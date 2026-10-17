"""Metrics, using the same averaging as Chinese MentalBERT: macro for EWECT/C2D2, binary for
Waimai/Suicide, micro for the multi-label Cognitive task. ``f1`` is always the task's
primary metric; macro-F1, accuracy and per-class F1 are always reported too."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SINGLE = "single_label_classification"
MULTI = "multi_label_classification"


def predictions_from_logits(logits: np.ndarray, problem_type: str) -> np.ndarray:
    if problem_type == MULTI:
        return (logits > 0).astype(int)  # sigmoid(x) > 0.5
    return logits.argmax(-1)


def classification_metrics(
    labels: np.ndarray, preds: np.ndarray, problem_type: str, average: str, label_names: list[str],
    labels_list: list[int] | None = None,
) -> dict:
    """`labels_list` restricts averaging to the real classes, so a prediction outside them (an
    unparsable LLM answer, coded -1) counts as wrong without becoming a class of its own."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    if average == "binary":  # anything but 1 (including -1) is a negative prediction
        p, r, f, _ = precision_recall_fscore_support(labels, np.where(preds == 1, 1, 0), average="binary",
                                                     pos_label=1, zero_division=0)
    else:
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average, labels=labels_list,
                                                     zero_division=0)
    out = {
        "f1": float(f),
        "precision": float(p),
        "recall": float(r),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=labels_list, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "average": average,
    }
    if problem_type == MULTI:
        per_class = f1_score(labels, preds, average=None, zero_division=0)
    else:
        per_class = f1_score(labels, preds, average=None, labels=list(range(len(label_names))), zero_division=0)
    out["per_class_f1"] = {name: float(v) for name, v in zip(label_names, per_class)}
    return out
