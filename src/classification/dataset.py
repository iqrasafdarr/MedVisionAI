"""Brain MRI classification dataset loader for MedVisionAI."""

from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import ViTImageProcessor


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


class BrainMRIDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        encoded = self.processor(
            images=image,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": label,
        }


def _collect_images(root_dir, class_names):
    root = Path(root_dir)
    samples = []

    class_to_idx = {
        _normalize_class_name(name): idx
        for idx, name in enumerate(class_names)
    }

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        normalized = _normalize_class_name(folder.name)

        if normalized not in class_to_idx:
            continue

        image_dir = folder / "images"

        if not image_dir.exists():
            continue

        label = class_to_idx[normalized]

        for image_path in sorted(image_dir.glob("*.jpg")):
            samples.append((str(image_path), label))

    return samples


def build_datasets(config: dict):
    data_cfg = config["data"]

    root_dir = Path(data_cfg["root_dir"])
    class_names = data_cfg["classes"]

    train_root = root_dir / "Train"
    test_root = root_dir / "Val"

    if not train_root.exists():
        raise FileNotFoundError(f"Training directory not found: {train_root}")

    if not test_root.exists():
        raise FileNotFoundError(f"Validation/test directory not found: {test_root}")

    all_train_samples = _collect_images(train_root, class_names)
    test_samples = _collect_images(test_root, class_names)

    if not all_train_samples:
        raise RuntimeError(f"No JPG images found in {train_root}")

    if not test_samples:
        raise RuntimeError(f"No JPG images found in {test_root}")

    labels = [label for _, label in all_train_samples]

    train_frac = data_cfg.get("train_frac", 0.9)
    val_frac = data_cfg.get("val_frac", 0.1)

    if abs(train_frac + val_frac - 1.0) > 1e-6:
        raise ValueError(
            "For the provided dataset split, train_frac + val_frac must equal 1.0."
        )

    train_samples, val_samples = train_test_split(
        all_train_samples,
        test_size=val_frac,
        stratify=labels,
        random_state=config["seed"],
    )

    processor = ViTImageProcessor.from_pretrained(
        config["model"]["name"]
    )

    train_dataset = BrainMRIDataset(train_samples, processor)
    val_dataset = BrainMRIDataset(val_samples, processor)
    test_dataset = BrainMRIDataset(test_samples, processor)

    print("\nClassification dataset:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Classes: {class_names}")

    return train_dataset, val_dataset, test_dataset, processor