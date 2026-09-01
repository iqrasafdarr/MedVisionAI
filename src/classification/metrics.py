"""Accuracy, precision, recall, macro F1, per-class metrics, and confusion
matrix computation for the classification module.

TODO(Phase 4): implement with sklearn.metrics on real model outputs only.
Also exposes get_confidence_buckets() for the trustworthy-AI reliability
analysis (correct-high-conf / incorrect-high-conf / low-conf), computed
from the ViT's actual softmax outputs on the test split.
"""


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    raise NotImplementedError("Implemented in Phase 4.")


def get_confidence_buckets(y_true, y_pred, y_proba, low_conf_threshold: float = 0.6) -> dict:
    """Bucket test-set predictions into correct-high-confidence,
    incorrect-high-confidence, and low-confidence groups for the
    trustworthy-AI section. Operates only on real model outputs."""
    raise NotImplementedError("Implemented in Phase 4.")
