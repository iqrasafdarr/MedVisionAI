"""ViT fine-tuning pipeline for MedVisionAI classification."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from src.classification.dataset import build_datasets
from src.classification.metrics import (
    compute_classification_metrics,
    get_confidence_buckets,
)


def _freeze_layers(model, num_layers):
    """Freeze ViT embeddings and the first N encoder layers."""

    if num_layers <= 0:
        return

    # Freeze patch embeddings.
    for parameter in model.vit.embeddings.parameters():
        parameter.requires_grad = False

    # Current Transformers version exposes ViT encoder blocks
    # through `model.vit.layers`.
    encoder_layers = model.vit.layers

    num_to_freeze = min(
        num_layers,
        len(encoder_layers),
    )

    for layer in encoder_layers[:num_to_freeze]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def _run_epoch(model, loader, criterion, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []
    all_proba = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(predictions.detach().cpu().numpy())
        all_proba.append(probabilities.detach().cpu().numpy())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_proba = np.concatenate(all_proba, axis=0)

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_proba,
    )

    metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics, y_true, y_pred, y_proba


def _save_confusion_matrix(matrix, class_names, output_path):
    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="ViT Brain MRI Classification Confusion Matrix",
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train(config: dict) -> dict:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    output_dir = Path(output_cfg["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = train_cfg.get("device", "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    print(f"\nUsing device: {device}")

    train_dataset, val_dataset, test_dataset, processor = build_datasets(
        config
    )

    num_workers = train_cfg.get("num_workers", 0)
    batch_size = train_cfg.get("batch_size", 2)

    train_loader = _make_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = _make_loader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = _make_loader(
        test_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("\nLoading pretrained ViT...")

    model = ViTForImageClassification.from_pretrained(
        model_cfg["name"],
        num_labels=model_cfg["num_labels"],
        ignore_mismatched_sizes=True,
    )

    model.to(device)

    freeze_layers = model_cfg.get("freeze_encoder_layers", 0)

    if freeze_layers > 0:
        _freeze_layers(model, freeze_layers)
        print(
            f"Frozen first {freeze_layers} ViT encoder layers "
            "for CPU-efficient fine-tuning."
        )

    class_names = data_cfg["classes"]

    train_labels = np.asarray(
        [label for _, label in train_dataset.samples]
    )

    class_weighting = train_cfg.get("class_weighting", "none")

    if class_weighting == "auto":
        counts = np.bincount(
            train_labels,
            minlength=len(class_names),
        ).astype(np.float32)

        weights = counts.sum() / (
            len(class_names) * np.maximum(counts, 1)
        )

        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        print("\nClass weights:")
        for name, weight in zip(class_names, weights):
            print(f"  {name}: {weight:.4f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    else:
        criterion = nn.CrossEntropyLoss()

    learning_rate = float(
        train_cfg.get("learning_rate", 2e-5)
    )
    weight_decay = float(
        train_cfg.get("weight_decay", 0.01)
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epochs = int(train_cfg.get("epochs", 3))

    log_file = Path(output_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1

    history = []

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        train_metrics, _, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        val_metrics, _, _, _ = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"train_f1={row['train_macro_f1']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_f1={row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            checkpoint_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": best_val_f1,
                    "class_names": class_names,
                    "model_name": model_cfg["name"],
                },
                checkpoint_path,
            )

            print(
                f"  Saved best checkpoint: {checkpoint_path}"
            )

    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nLoading best checkpoint for final test evaluation...")

    checkpoint_path = output_dir / "best_model.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_metrics, y_true, y_pred, y_proba = _run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        training=False,
    )

    confidence = get_confidence_buckets(
        y_true,
        y_pred,
        y_proba,
        low_conf_threshold=0.6,
    )

    confusion_matrix_path = Path(
        output_cfg["confusion_matrix_file"]
    )

    _save_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        confusion_matrix_path,
    )

    predictions_path = (
        output_dir.parent / "test_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    summary = {
        "model": model_cfg["name"],
        "device": str(device),
        "num_classes": len(class_names),
        "classes": class_names,
        "dataset": {
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "test_samples": len(test_dataset),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
            "freeze_encoder_layers": freeze_layers,
        },
        "test_metrics": test_metrics,
        "confidence_analysis": confidence,
    }

    summary_path = Path(output_cfg["summary_file"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal test results:")
    print(
        f"  Accuracy:        "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:        "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Macro Precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )
    print(
        f"  Macro Recall:    "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print("\nConfidence analysis:")
    print(
        f"  Low confidence: "
        f"{confidence['low_confidence']['count']}"
    )
    print(
        f"  Correct high confidence: "
        f"{confidence['correct_high_confidence']['count']}"
    )
    print(
        f"  Incorrect high confidence: "
        f"{confidence['incorrect_high_confidence']['count']}"
    )

    return summary