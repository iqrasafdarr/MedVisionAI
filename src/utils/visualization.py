"""Shared plotting helper for training curves, used by any module that logs
per-epoch metrics via src/utils/logging.py::append_epoch_log (currently the
segmentation module; the classification module will reuse this in Phase 4).

Prediction-overlay visualization is NOT here — it's implemented directly in
src/segmentation/visualize.py, since overlay rendering needs
segmentation-specific context (image/mask/prediction arrays, Dice-based
sampling) that doesn't generalize cleanly as a shared helper.
"""

import csv

import matplotlib.pyplot as plt


def plot_training_curves(log_csv_path: str, out_path: str) -> None:
    """Plot train/val loss (and val_dice, if present) from an epoch log CSV
    written by src/utils/logging.py::append_epoch_log.

    Rows where val_loss/val_dice are empty (epochs skipped by
    train.val_frequency) are omitted from those lines rather than plotted
    as zero.
    """
    epochs, train_loss, val_loss, val_dice = [], [], [], []
    with open(log_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]) if row.get("train_loss") else None)
            val_loss.append(float(row["val_loss"]) if row.get("val_loss") else None)
            val_dice.append(float(row["val_dice"]) if row.get("val_dice") else None)

    if not epochs:
        raise ValueError(f"No rows found in '{log_csv_path}' — nothing to plot.")

    def _valid_pairs(y_values):
        return [(e, y) for e, y in zip(epochs, y_values) if y is not None]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    tl_pairs = _valid_pairs(train_loss)
    vl_pairs = _valid_pairs(val_loss)
    axes[0].plot(*zip(*tl_pairs), label="train_loss")
    if vl_pairs:
        axes[0].plot(*zip(*vl_pairs), label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    vd_pairs = _valid_pairs(val_dice)
    if vd_pairs:
        axes[1].plot(*zip(*vd_pairs), color="green", label="val_dice")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("Dice")
        axes[1].set_title("Validation Dice")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
