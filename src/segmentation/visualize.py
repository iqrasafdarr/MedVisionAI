"""Input MRI -> Ground Truth Mask -> Predicted Mask -> Overlay visualization
for representative test cases.

Cases are sampled across the per-slice Dice distribution (best, median,
worst-scoring foreground-containing slices) rather than only cherry-picked
best results, so the figure set is honest about failure cases too — this is
directly what the trustworthy-AI framing of the project asks for.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

from src.segmentation.dataset import build_dataloaders
from src.segmentation.evaluate import load_checkpoint
from src.segmentation.model import get_device


def _save_case_figure(image, gt_mask, pred_mask, dice_score, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input MRI (FLAIR)")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap="autumn", alpha=0.6)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="cool", alpha=0.6)
    axes[2].set_title("Prediction")

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap="autumn", alpha=0.4)
    axes[3].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="cool", alpha=0.4)
    axes[3].set_title(f"Overlay (Dice={dice_score:.3f})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_predictions(config: dict, checkpoint_path: str | None = None,
                           n_examples: int = 8) -> list[str]:
    """Generate MRI / GT / Prediction / Overlay figures for representative
    test-set slices, sampled across the Dice-score distribution.

    Returns:
        List of saved figure paths.
    """
    output_cfg = config["output"]
    device = get_device(config["train"].get("device", "cuda"))

    checkpoint_path = checkpoint_path or str(
        Path(output_cfg["checkpoint_dir"]) / "best_model.pt"
    )
    model, _checkpoint = load_checkpoint(checkpoint_path, device)

    _train_loader, _val_loader, test_loader, _split = build_dataloaders(config)
    dice_metric = DiceMetric(include_background=False, reduction="none")

    scored_cases = []  # (dice, image_np, gt_np, pred_np)
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1, keepdim=True)

            labels_onehot = one_hot(labels.unsqueeze(1), num_classes=logits.shape[1])
            preds_onehot = one_hot(preds, num_classes=logits.shape[1])
            batch_dice = dice_metric(y_pred=preds_onehot, y=labels_onehot).flatten()

            for i in range(images.shape[0]):
                dice_val = batch_dice[i].item()
                if np.isnan(dice_val):
                    continue  # skip slices with no foreground in GT or pred
                scored_cases.append(
                    (
                        dice_val,
                        images[i, 0].cpu().numpy(),
                        labels[i].cpu().numpy(),
                        preds[i, 0].cpu().numpy(),
                    )
                )

    if not scored_cases:
        raise RuntimeError(
            "No foreground-containing test slices were found to visualize. "
            "This can happen with a very small/synthetic smoke-test dataset."
        )

    scored_cases.sort(key=lambda c: c[0])
    n = len(scored_cases)
    # sample across the distribution: worst, low-mid, median, high-mid, best
    sample_positions = sorted(
        set(
            max(0, min(n - 1, int(round(p * (n - 1)))))
            for p in np.linspace(0, 1, min(n_examples, n))
        )
    )

    out_dir = Path(output_cfg["visualization_dir"])
    saved_paths = []
    for rank, pos in enumerate(sample_positions):
        dice_val, image_np, gt_np, pred_np = scored_cases[pos]
        out_path = out_dir / f"test_case_{rank:02d}_dice_{dice_val:.3f}.png"
        _save_case_figure(image_np, gt_np, pred_np, dice_val, out_path)
        saved_paths.append(str(out_path))

    return saved_paths
