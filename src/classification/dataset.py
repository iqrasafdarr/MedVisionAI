"""Loading for the existing 4-class brain tumor MRI dataset, ViT-compatible
preprocessing, label encoding, and stratified train/val/test split.

TODO(Phase 4):
- Load from configs/classification.yaml data.root_dir (local path, not committed)
- Stratified split (class-balanced across train/val/test)
- ViTImageProcessor-compatible preprocessing
- Report class distribution for the class_weighting="auto" handling
"""


def build_datasets(config: dict):
    raise NotImplementedError("Implemented in Phase 4.")
