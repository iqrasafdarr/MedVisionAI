"""MSD Task01_BrainTumour dataset handling for the segmentation module.

Label strategy (documented explicitly, per project requirements)
------------------------------------------------------------------
The Medical Segmentation Decathlon Task01_BrainTumour labels use:
    0 = background
    1 = edema
    2 = non-enhancing tumor
    3 = enhancing tumor
This project performs **binary** tumor segmentation. All non-zero labels
(1, 2, 3) are collapsed into a single foreground class:
    0 = background
    1 = tumor (edema + non-enhancing + enhancing, combined)
This is a deliberate simplification (documented in docs/experiments.md) —
multi-class tumor sub-region segmentation is out of scope for this phase.

Modality strategy
------------------
MSD Task01 images are 4-channel volumes stacked in the order
[FLAIR, T1w, T1gd, T2w] (per the dataset's own dataset.json). This project
uses **FLAIR only** (channel index 0), consistent with configs/segmentation.yaml.

Patient-level splitting
------------------------
Each file in imagesTr/labelsTr corresponds to one patient/volume. The split
is performed at the **volume level** — every slice extracted from a given
volume goes entirely into train, val, or test, never split across them.
The resulting split is persisted to `data.split_file` so it's inspectable
and reused across runs (not re-randomized every run).

2D slice extraction
---------------------
Full 3D U-Net training is out of scope for this project's Colab compute
budget (documented in docs/limitations.md). Instead, each 3D volume is
decomposed into 2D axial slices. Training/validation slices that are
entirely background (no brain tissue) are dropped — they carry no learning
signal. Test slices keep the full distribution, including background-only
slices, so evaluation reflects real-world slice composition.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetIntegrityError(RuntimeError):
    """Raised when the MSD dataset directory does not have the expected
    structure or files are missing. Never caught silently — the pipeline
    must fail loudly rather than continue with missing data."""


@dataclass
class VolumeRecord:
    patient_id: str
    image_path: Path
    label_path: Path


def discover_volumes(root_dir: str) -> list[VolumeRecord]:
    """Scan an MSD Task01_BrainTumour directory and pair each image with its
    label by filename.

    Expected structure (the standard MSD layout):
        root_dir/
            imagesTr/BRATS_XXX.nii.gz
            labelsTr/BRATS_XXX.nii.gz

    Raises:
        DatasetIntegrityError: if the directory, images, or labels are
        missing, or if an image has no matching label.
    """
    root = Path(root_dir)
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"

    if not root.exists():
        raise DatasetIntegrityError(
            f"Dataset root '{root}' does not exist. Run "
            f"`python scripts/download_data.py --config configs/segmentation.yaml` "
            f"first, or check configs/segmentation.yaml: data.root_dir."
        )
    if not images_dir.exists() or not labels_dir.exists():
        raise DatasetIntegrityError(
            f"Expected '{images_dir}' and '{labels_dir}' to exist. The MSD "
            f"Task01_BrainTumour layout requires both imagesTr/ and labelsTr/ "
            f"under the dataset root. Found root contents: "
            f"{[p.name for p in root.iterdir()] if root.exists() else 'N/A'}"
        )

    image_files = sorted(
        p for p in images_dir.glob("*.nii.gz") if not p.name.startswith(".")
    )
    if not image_files:
        raise DatasetIntegrityError(
            f"No .nii.gz images found under '{images_dir}'. The dataset "
            f"appears to be empty or incorrectly extracted."
        )

    records: list[VolumeRecord] = []
    missing_labels = []
    for image_path in image_files:
        label_path = labels_dir / image_path.name
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        records.append(
            VolumeRecord(
                patient_id=image_path.name.replace(".nii.gz", ""),
                image_path=image_path,
                label_path=label_path,
            )
        )

    if missing_labels:
        raise DatasetIntegrityError(
            f"{len(missing_labels)} image(s) have no matching label file in "
            f"'{labels_dir}': {missing_labels[:5]}"
            f"{' ...' if len(missing_labels) > 5 else ''}. Refusing to "
            f"proceed with an incomplete dataset."
        )

    return records


def get_patient_level_split(
    root_dir: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    split_file: str,
) -> dict[str, list[str]]:
    """Compute (or load, if already computed) a patient/volume-ID-keyed
    train/val/test split and persist it to `split_file`.

    Splitting is done by shuffling the list of patient IDs (not slices) with
    a fixed seed, then partitioning by the given fractions. Re-running with
    the same config reuses the persisted split file rather than
    re-randomizing, so results stay comparable across runs.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test fractions must sum to 1.0, got "
            f"{train_frac} + {val_frac} + {test_frac} = "
            f"{train_frac + val_frac + test_frac}"
        )

    split_path = Path(split_file)
    if split_path.exists():
        with split_path.open() as f:
            existing = json.load(f)
        return {k: existing[k] for k in ("train", "val", "test")}

    records = discover_volumes(root_dir)
    patient_ids = sorted({r.patient_id for r in records})

    rng = random.Random(seed)
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # test gets the remainder, so rounding never drops/duplicates a patient
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    split = {
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "n_total_patients": n,
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w") as f:
        json.dump(split, f, indent=2)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _extract_flair_channel(image_4ch: np.ndarray) -> np.ndarray:
    """MSD Task01 images are stored as (H, W, D, 4) with channel order
    [FLAIR, T1w, T1gd, T2w]. Return the FLAIR channel only, shape (H, W, D)."""
    if image_4ch.ndim != 4 or image_4ch.shape[-1] != 4:
        raise DatasetIntegrityError(
            f"Expected a 4-channel MSD Task01 volume with shape (H, W, D, 4), "
            f"got shape {image_4ch.shape}. This does not look like a valid "
            f"Task01_BrainTumour image."
        )
    return image_4ch[..., 0]


def _binarize_label(label: np.ndarray) -> np.ndarray:
    """Collapse labels {1, 2, 3} -> 1 (tumor), keep 0 as background."""
    return (label > 0).astype(np.uint8)


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Per-slice z-score normalization over non-zero (brain-tissue) voxels,
    since background is a hard zero and would otherwise dominate the
    mean/std of the whole slice."""
    mask = slice_2d > 0
    if mask.sum() == 0:
        return slice_2d.astype(np.float32)
    mean = slice_2d[mask].mean()
    std = slice_2d[mask].std()
    std = std if std > 1e-8 else 1.0
    normalized = (slice_2d - mean) / std
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def _resize_2d(array_2d: np.ndarray, target_size: tuple[int, int], is_label: bool) -> np.ndarray:
    """Resize a 2D array to target_size. Nearest-neighbor for labels
    (preserves discrete class values), bilinear for images."""
    tensor = torch.from_numpy(array_2d).float().unsqueeze(0).unsqueeze(0)
    mode = "nearest" if is_label else "bilinear"
    kwargs = {} if is_label else {"align_corners": False}
    resized = F.interpolate(tensor, size=target_size, mode=mode, **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()


def _build_slice_index(
    records: list[VolumeRecord], patient_ids: set[str], drop_empty_slices: bool
) -> list[tuple[Path, Path, int]]:
    """Scan the relevant volumes once and build a flat (image_path,
    label_path, slice_index) index, optionally dropping slices with no
    brain tissue (all-background) in the image.

    Done eagerly at dataset construction (not lazily per __getitem__) so
    __len__ is well-defined and slice filtering is transparent.
    """
    index: list[tuple[Path, Path, int]] = []
    for record in records:
        if record.patient_id not in patient_ids:
            continue
        image_4ch = nib.load(str(record.image_path)).get_fdata()
        image_flair = _extract_flair_channel(image_4ch)

        num_slices = image_flair.shape[2]
        for slice_idx in range(num_slices):
            if drop_empty_slices:
                image_slice = image_flair[:, :, slice_idx]
                if image_slice.max() <= 0:
                    continue  # no brain tissue in this slice at all
            index.append((record.image_path, record.label_path, slice_idx))
    return index


class BrainTumorSliceDataset(Dataset):
    """2D axial-slice dataset over MSD Task01_BrainTumour FLAIR volumes.

    Each item is one (image_slice, label_slice) pair, resized to
    `patch_size` and normalized. Slices are indexed eagerly at construction
    (see `_build_slice_index`) so `len(dataset)` is accurate.
    """

    def __init__(
        self,
        records: list[VolumeRecord],
        patient_ids: list[str],
        patch_size: tuple[int, int],
        drop_empty_slices: bool = True,
    ) -> None:
        self.patch_size = tuple(patch_size)
        patient_id_set = set(patient_ids)
        self.index = _build_slice_index(
            records, patient_id_set, drop_empty_slices=drop_empty_slices
        )
        if not self.index:
            raise DatasetIntegrityError(
                "No usable slices found for the requested patient split. "
                "Check that the dataset was downloaded correctly and that "
                "the split file is not empty."
            )
        # cache the most recently loaded volume, since consecutive indices
        # usually belong to the same volume
        self._cache_path: Path | None = None
        self._cache_image: np.ndarray | None = None
        self._cache_label: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_volume(self, image_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_path == image_path:
            return self._cache_image, self._cache_label
        image_4ch = nib.load(str(image_path)).get_fdata()
        image_flair = _extract_flair_channel(image_4ch)
        label = nib.load(str(label_path)).get_fdata()
        self._cache_path = image_path
        self._cache_image = image_flair
        self._cache_label = label
        return image_flair, label

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, label_path, slice_idx = self.index[idx]
        image_flair, label = self._load_volume(image_path, label_path)

        image_slice = image_flair[:, :, slice_idx]
        label_slice = _binarize_label(label[:, :, slice_idx])

        image_slice = _normalize_slice(image_slice)
        image_resized = _resize_2d(image_slice, self.patch_size, is_label=False)
        label_resized = _resize_2d(
            label_slice.astype(np.float32), self.patch_size, is_label=True
        )

        image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)  # (1, H, W)
        label_tensor = torch.from_numpy(label_resized).long()  # (H, W)

        return {"image": image_tensor, "label": label_tensor}


def build_dataloaders(config: dict):
    """Build train/val/test DataLoaders from a resolved segmentation config.

    Returns:
        (train_loader, val_loader, test_loader, split) tuple.
    """
    data_cfg = config["data"]
    records = discover_volumes(data_cfg["root_dir"])
    split = get_patient_level_split(
        root_dir=data_cfg["root_dir"],
        train_frac=data_cfg["train_frac"],
        val_frac=data_cfg["val_frac"],
        test_frac=data_cfg["test_frac"],
        seed=config["seed"],
        split_file=data_cfg["split_file"],
    )

    patch_size = tuple(data_cfg["patch_size"])
    train_ds = BrainTumorSliceDataset(records, split["train"], patch_size, drop_empty_slices=True)
    val_ds = BrainTumorSliceDataset(records, split["val"], patch_size, drop_empty_slices=True)
    # Test keeps empty slices too — evaluation should reflect real-world
    # slice distribution, not just tissue-containing slices.
    test_ds = BrainTumorSliceDataset(records, split["test"], patch_size, drop_empty_slices=False)

    train_cfg = config["train"]
    num_workers = train_cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, split
