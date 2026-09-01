"""Segmentation training loop.

Loss choice: DiceCELoss (MONAI) — a weighted sum of Dice loss and
cross-entropy. Dice loss alone can produce unstable gradients early in
training when the foreground (tumor) class is small relative to background;
combining it with cross-entropy gives more stable per-pixel gradients while
Dice keeps the optimization aligned with the metric we actually report
(Dice score). This is a standard, well-documented choice for medical
segmentation with class imbalance, not a novel contribution of this project.

Model selection: the checkpoint with the highest validation Dice is kept as
`best_model.pt` (not the lowest validation loss), since Dice is the metric
this project reports and optimizes for.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

from src.segmentation.dataset import build_dataloaders
from src.segmentation.model import build_model, get_device
from src.utils.logging import append_epoch_log, write_summary


def _run_validation(model, val_loader, device, loss_fn, dice_metric) -> tuple[float, float]:
    model.eval()
    dice_metric.reset()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            labels_onehot = one_hot(labels.unsqueeze(1), num_classes=logits.shape[1])
            loss = loss_fn(logits, labels_onehot)
            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1, keepdim=True)
            preds_onehot = one_hot(preds, num_classes=logits.shape[1])
            dice_metric(y_pred=preds_onehot, y=labels_onehot)

    mean_loss = total_loss / max(n_batches, 1)
    mean_dice = dice_metric.aggregate().item()
    return mean_loss, mean_dice


def train(config: dict) -> dict:
    """Run segmentation training end to end: build data/model, train with
    early stopping on validation Dice, checkpoint the best model, log every
    epoch, and write a final summary.

    Returns:
        Summary dict (also written to output.summary_file).
    """
    train_cfg = config["train"]
    output_cfg = config["output"]
    device = get_device(train_cfg.get("device", "cuda"))

    train_loader, val_loader, _test_loader, split = build_dataloaders(config)

    model = build_model(config["model"], spatial_dims=2).to(device)

    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg["learning_rate"])
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    best_val_dice = -1.0
    epochs_without_improvement = 0
    patience = train_cfg.get("early_stopping_patience", 10)
    max_epochs = train_cfg["epochs"]
    val_frequency = max(1, train_cfg.get("val_frequency", 1))

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            labels_onehot = one_hot(labels.unsqueeze(1), num_classes=logits.shape[1])
            loss = loss_fn(logits, labels_onehot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        if epoch % val_frequency != 0 and epoch != max_epochs:
            append_epoch_log(
                output_cfg["log_file"],
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": None,
                    "val_dice": None,
                    "epoch_time_sec": round(epoch_time, 2),
                },
            )
            print(f"[epoch {epoch}/{max_epochs}] train_loss={train_loss:.4f} (no val this epoch)")
            continue

        val_loss, val_dice = _run_validation(model, val_loader, device, loss_fn, dice_metric)

        append_epoch_log(
            output_cfg["log_file"],
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_dice": round(val_dice, 6),
                "epoch_time_sec": round(epoch_time, 2),
            },
        )
        print(
            f"[epoch {epoch}/{max_epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"({epoch_time:.1f}s)"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "model_config": config["model"],
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val_dice improvement for {patience} validated checks)."
                )
                break

    summary = {
        "seed": config["seed"],
        "device": str(device),
        "epochs_run": epoch,
        "max_epochs_configured": max_epochs,
        "best_val_dice": best_val_dice,
        "best_checkpoint_path": str(best_checkpoint_path),
        "model_config": config["model"],
        "train_config": train_cfg,
        "split_sizes": {k: len(v) for k, v in split.items()},
    }
    write_summary(output_cfg["summary_file"], summary)
    return summary
