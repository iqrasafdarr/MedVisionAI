"""Vision Transformer fine-tuning pipeline for brain MRI classification."""

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


def _make_loader(dataset, batch_size, shuffle, num_workers):
    """Create a DataLoader with CPU-friendly settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def _freeze_layers(model, num_layers):
    """Freeze ViT embeddings and the first N encoder layers."""

    if num_layers <= 0:
        return

    # Freeze patch embeddings.
    for parameter in model.vit.embeddings.parameters():
        parameter.requires_grad = False

    # Current Transformers ViT implementation exposes
    # encoder blocks through `model.vit.layers`.
    encoder_layers = model.vit.layers

    num_to_freeze = min(
        num_layers,
        len(encoder_layers),
    )

    for layer in encoder_layers[:num_to_freeze]:
        for parameter in layer.parameters():
            parameter.requires_grad = False

    print(
        f"Frozen first {num_to_freeze} ViT encoder layers "
        f"for CPU-efficient fine-tuning."
    )


def _run_epoch(model, loader, criterion, optimizer, device, training):
    """Run one training or validation epoch."""

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            if training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        total_loss += loss.item() * labels.size(0)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        all_labels.extend(
            labels.detach().cpu().numpy()
        )
        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

    mean_loss = total_loss / len(loader.dataset)

    return (
        mean_loss,
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def _save_confusion_matrix(
    confusion_matrix,
    class_names,
    output_path,
):
    """Save confusion matrix visualization."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.imshow(confusion_matrix)

    ax.set_title("ViT Brain MRI Classification Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(
        range(len(class_names))
    )
    ax.set_yticks(
        range(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(confusion_matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def train(config):
    """Train and evaluate the ViT classification model."""

    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device_name = config["train"].get(
        "device",
        "cpu",
    )

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    train_dataset, val_dataset, test_dataset, class_names = build_datasets(
        config
    )

    print("\nClassification dataset:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Classes: {class_names}")

    num_workers = config["train"].get(
        "num_workers",
        0,
    )

    batch_size = config["train"].get(
        "batch_size",
        2,
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = _make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = _make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    print("\nLoading pretrained ViT...")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    model.config.id2label = {
        i: class_name
        for i, class_name in enumerate(class_names)
    }

    model.config.label2id = {
        class_name: i
        for i, class_name in enumerate(class_names)
    }

    model.to(device)

    freeze_layers = config["model"].get(
        "freeze_encoder_layers",
        0,
    )

    _freeze_layers(
        model,
        freeze_layers,
    )

    # ------------------------------------------------------------------
    # Class weighting
    # ------------------------------------------------------------------

    class_weighting = config["train"].get(
        "class_weighting",
        "none",
    )

    criterion = None

    if class_weighting == "auto":
        train_labels = [
            label
            for _, label in train_dataset.samples
        ]

        class_counts = np.bincount(
            train_labels,
            minlength=num_labels,
        )

        total_samples = len(train_labels)

        weights = total_samples / (
            num_labels * np.maximum(class_counts, 1)
        )

        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print("\nClass weights:")

        for class_name, weight in zip(
            class_names,
            weights,
        ):
            print(
                f"  {class_name}: {weight:.4f}"
            )

    else:
        criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    learning_rate = float(
        config["train"].get(
            "learning_rate",
            2e-5,
        )
    )

    weight_decay = float(
        config["train"].get(
            "weight_decay",
            0.01,
        )
    )

    optimizer = AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epochs = int(
        config["train"].get(
            "epochs",
            3,
        )
    )

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------

    checkpoint_dir = Path(
        config["output"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = Path(
        config["output"]["log_file"]
    )

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = Path(
        config["output"]["summary_file"]
    )

    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrix_file = Path(
        config["output"]["confusion_matrix_file"]
    )

    confusion_matrix_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        checkpoint_dir / "best_model.pt"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print(
        f"\nStarting training for {epochs} epochs..."
    )

    best_val_f1 = -float("inf")
    training_history = []

    for epoch in range(1, epochs + 1):

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        train_loss, train_labels, train_predictions, train_probs = (
            _run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=True,
            )
        )

        val_loss, val_labels, val_predictions, val_probs = (
            _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=False,
            )
        )

        train_metrics = compute_classification_metrics(
            train_labels,
            train_predictions,
            train_probs,
            class_names,
        )

        val_metrics = compute_classification_metrics(
            val_labels,
            val_predictions,
            val_probs,
            class_names,
        )

        print(
            f"  Train loss: {train_loss:.4f}"
        )
        print(
            f"  Train accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Train macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Val loss: {val_loss:.4f}"
        )
        print(
            f"  Val accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        training_history.append(
            epoch_record
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": best_val_f1,
                    "class_names": class_names,
                    "model_name": model_name,
                },
                best_checkpoint,
            )

            print(
                f"  Saved best checkpoint: "
                f"{best_checkpoint}"
            )

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------

    with open(
        log_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(
            training_history
        )

    # ------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------

    print(
        "\nLoading best checkpoint for test evaluation..."
    )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    test_loss, test_labels, test_predictions, test_probs = (
        _run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )
    )

    test_metrics = compute_classification_metrics(
        test_labels,
        test_predictions,
        test_probs,
        class_names,
    )

    confidence = get_confidence_buckets(
        test_labels,
        test_predictions,
        test_probs,
    )

    print("\nTest results:")
    print(
        f"  Loss: {test_loss:.4f}"
    )
    print(
        f"  Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"  Macro precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )
    print(
        f"  Macro recall: "
        f"{test_metrics['macro_recall']:.4f}"
    )
    print(
        f"  Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Mean confidence: "
        f"{test_metrics['mean_confidence']:.4f}"
    )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    _save_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        confusion_matrix_file,
    )

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------

    predictions_file = (
        Path("results/classification")
        / "test_predictions.npz"
    )

    np.savez_compressed(
        predictions_file,
        labels=test_labels,
        predictions=test_predictions,
        probabilities=test_probs,
    )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------

    summary = {
        "model_name": model_name,
        "num_classes": num_labels,
        "class_names": class_names,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_checkpoint": str(best_checkpoint),
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "confidence_analysis": confidence,
        "training_history": training_history,
    }

    # Convert NumPy values to JSON-safe values.
    def make_json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, dict):
            return {
                key: make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                make_json_safe(item)
                for item in value
            ]

        return value

    summary = make_json_safe(summary)

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("\nClassification experiment complete.")

    print(
        f"  Best checkpoint: {best_checkpoint}"
    )
    print(
        f"  Training log: {log_file}"
    )
    print(
        f"  Summary: {summary_file}"
    )
    print(
        f"  Confusion matrix: "
        f"{confusion_matrix_file}"
    )
    print(
        f"  Predictions: "
        f"{predictions_file}"
    )