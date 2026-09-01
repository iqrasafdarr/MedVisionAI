"""Download (or verify) the Medical Segmentation Decathlon Task01_BrainTumour
dataset for the segmentation module.

Usage:
    python scripts/download_data.py --config configs/segmentation.yaml
    python scripts/download_data.py --config configs/segmentation.yaml --verify-only

Automatic download uses MONAI's monai.apps.DecathlonDataset, which fetches
the official Task01_BrainTumour archive from the Decathlon organizers'
Google Drive-hosted release and extracts it to the configured root_dir.
This requires outbound internet access to Google Drive, which is available
on Google Colab and most local machines, but is NOT available in every
sandboxed environment (including the one used to build this repository —
see the Phase 3 implementation notes for what was and wasn't actually run).

If the automatic download fails (a known occasional issue with Google
Drive's download quotas), download manually instead:

    1. Go to http://medicaldecathlon.com/
    2. Under "Download", locate Task01_BrainTumour and download
       Task01_BrainTumour.tar
    3. Extract it so the resulting structure matches:
           <root_dir>/imagesTr/*.nii.gz
           <root_dir>/labelsTr/*.nii.gz
       (this is the archive's native layout — no reorganizing needed)
    4. Re-run this script with --verify-only to confirm the structure.

The dataset is provided by the Medical Segmentation Decathlon for research
use — check http://medicaldecathlon.com for current licensing terms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repository root is importable regardless of the working
# directory this script is invoked from (`python scripts/download_data.py`
# only puts scripts/ on sys.path by default, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.segmentation.dataset import DatasetIntegrityError, discover_volumes


def verify_dataset(root_dir: str) -> None:
    """Verify the dataset directory has the expected imagesTr/labelsTr
    structure with matched image/label pairs. Raises DatasetIntegrityError
    with an actionable message if anything is wrong."""
    records = discover_volumes(root_dir)
    print(f"Verified {len(records)} matched image/label volume pairs under '{root_dir}'.")


def download_dataset(root_dir: str) -> None:
    """Download Task01_BrainTumour via MONAI's DecathlonDataset into
    root_dir's parent, then verify the resulting structure.

    MONAI's DecathlonDataset downloads+extracts to
    `<download_dir>/Task01_BrainTumour/`, so root_dir should point directly
    at that extracted folder (as configured in configs/segmentation.yaml).
    """
    try:
        from monai.apps import DecathlonDataset
    except ImportError as e:
        raise RuntimeError(
            "monai is required for automatic download. Install it with "
            "`pip install monai` (already in requirements.txt)."
        ) from e

    root = Path(root_dir)
    download_dir = root.parent
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Task01_BrainTumour to '{download_dir}' via MONAI DecathlonDataset...")
    print("This is a multi-GB download and may take a while.")
    try:
        # download=True triggers the fetch+extract; we only need the side
        # effect (files on disk), not the returned MONAI Dataset object,
        # since this project's own dataset/dataloading logic
        # (src/segmentation/dataset.py) reads the extracted files directly.
        DecathlonDataset(
            root_dir=str(download_dir),
            task="Task01_BrainTumour",
            section="training",
            download=True,
            cache_num=0,  # do not pre-cache transformed tensors; we only want the raw files
            num_workers=0,
        )
    except Exception as e:  # noqa: BLE001 - surface the real error, then guidance
        print(
            f"\nAutomatic download failed: {e}\n\n"
            f"This can happen due to Google Drive download quotas. "
            f"Follow the manual download steps in this script's module "
            f"docstring, then re-run with --verify-only.",
            file=sys.stderr,
        )
        raise

    verify_dataset(root_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip downloading; only check that the dataset directory is correctly structured.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    root_dir = config["data"]["root_dir"]

    if args.verify_only:
        verify_dataset(root_dir)
        return

    if Path(root_dir).exists():
        try:
            verify_dataset(root_dir)
            print("Dataset already present and verified — skipping download.")
            return
        except DatasetIntegrityError:
            print(f"'{root_dir}' exists but is incomplete/invalid — attempting download.")

    download_dataset(root_dir)


if __name__ == "__main__":
    main()
