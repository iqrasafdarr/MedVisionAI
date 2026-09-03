"""Classification metrics and confidence analysis for MedVisionAI."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        metrics["mean_confidence"] = float(np.max(y_proba, axis=1).mean())

    return metrics


def get_confidence_buckets(
    y_true,
    y_pred,
    y_proba,
    low_conf_threshold: float = 0.6,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    confidence = np.max(y_proba, axis=1)

    low_conf = confidence < low_conf_threshold
    high_conf = ~low_conf

    correct_high = high_conf & (y_true == y_pred)
    incorrect_high = high_conf & (y_true != y_pred)

    total = len(y_true)

    return {
        "low_confidence": {
            "count": int(low_conf.sum()),
            "percentage": float(low_conf.sum() / total * 100),
        },
        "correct_high_confidence": {
            "count": int(correct_high.sum()),
            "percentage": float(correct_high.sum() / total * 100),
        },
        "incorrect_high_confidence": {
            "count": int(incorrect_high.sum()),
            "percentage": float(incorrect_high.sum() / total * 100),
        },
        "threshold": float(low_conf_threshold),
        "mean_confidence": float(confidence.mean()),
    }