"""Segmentation evaluation on the held-out (patient-level) test split.

Reports per-slice Dice and IoU (Jaccard), then aggregates to mean and
standard deviation across test slices. This is a slice-level evaluation
(consistent with the 2D-slice training pipeline), not a reconstructed
3D-volume Dice — that distinction is documented here and in
docs/experiments.md so results aren't misread as full-volume metrics.
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.utils import one_hot

from src.segmentation.dataset import build_dataloaders
from src.segmentation.model import build_model, get_device
from src.utils.logging import write_summary


def load_checkpoint(checkpoint_path: str, device: torch.device):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Train the model first with "
            f"`python scripts/train_segmentation.py --config configs/segmentation.yaml`."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_config"], spatial_dims=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(config: dict, checkpoint_path: str | None = None) -> dict:
    """Evaluate a trained checkpoint on the test split.

    Returns:
        dict with per-slice mean/std Dice and IoU, plus n_test_slices,
        written to output.summary_file's sibling `segmentation_metrics.json`.
    """
    output_cfg = config["output"]
    device = get_device(config["train"].get("device", "cuda"))

    checkpoint_path = checkpoint_path or str(
        Path(output_cfg["checkpoint_dir"]) / "best_model.pt"
    )
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    _train_loader, _val_loader, test_loader, split = build_dataloaders(config)

    dice_metric = DiceMetric(include_background=False, reduction="none")
    iou_metric = MeanIoU(include_background=False, reduction="none")

    all_dice: list[float] = []
    all_iou: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1, keepdim=True)

            labels_onehot = one_hot(labels.unsqueeze(1), num_classes=logits.shape[1])
            preds_onehot = one_hot(preds, num_classes=logits.shape[1])

            batch_dice = dice_metric(y_pred=preds_onehot, y=labels_onehot)
            batch_iou = iou_metric(y_pred=preds_onehot, y=labels_onehot)

            all_dice.extend(batch_dice.flatten().cpu().tolist())
            all_iou.extend(batch_iou.flatten().cpu().tolist())

    dice_tensor = torch.tensor(all_dice)
    iou_tensor = torch.tensor(all_iou)
    # Slices with no foreground in both pred and label produce NaN Dice/IoU
    # under MONAI's default; exclude those from the aggregate rather than
    # silently treating them as 0 or 1, and report how many were excluded.
    valid_dice = dice_tensor[~torch.isnan(dice_tensor)]
    valid_iou = iou_tensor[~torch.isnan(iou_tensor)]

    metrics = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_dice": checkpoint.get("val_dice"),
        "n_test_slices_total": len(all_dice),
        "n_test_slices_scored": int(valid_dice.numel()),
        "n_test_slices_excluded_no_foreground": int(len(all_dice) - valid_dice.numel()),
        "dice_mean": float(valid_dice.mean()) if valid_dice.numel() else None,
        "dice_std": float(valid_dice.std()) if valid_dice.numel() > 1 else None,
        "iou_mean": float(valid_iou.mean()) if valid_iou.numel() else None,
        "iou_std": float(valid_iou.std()) if valid_iou.numel() > 1 else None,
        "test_patient_ids": split["test"],
    }

    metrics_path = str(Path(output_cfg["summary_file"]).parent / "segmentation_metrics.json")
    write_summary(metrics_path, metrics)
    return metrics
