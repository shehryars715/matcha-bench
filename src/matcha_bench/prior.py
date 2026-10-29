"""Class-prior correction of a model's scores, tuned on dev (an analysis of the baselines).

The post-level training sets are capped at 5,000 posts per class and most models train with balanced
class weights, while dev and test keep the natural class mix (61% DEP on disorder_post test), so a model
over-predicts the rare classes. The standard correction (Saerens et al., 2002; Menon et al., 2021) adds
tau * (log pi_dev - log pi_train) to its log-scores: pi_train is the class mix the model was trained
toward (uniform under balanced class weights, else the training-set mix) and pi_dev the dev label mix.
tau = 1 is the Bayes correction for accuracy; for macro-F1, tau is picked on dev (one number per run).
"""

from __future__ import annotations

import numpy as np

from .finetune.metrics import classification_metrics


def class_mix(labels, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=n_classes) / len(labels)


def log_shift(dev_labels, n_classes: int, train_mix: np.ndarray | None = None) -> np.ndarray:
    """log pi_dev - log pi_train per class; `train_mix` None means uniform (balanced class weights)."""
    train = np.full(n_classes, 1.0 / n_classes) if train_mix is None else np.asarray(train_mix, dtype=float)
    return np.log(np.maximum(class_mix(dev_labels, n_classes), 1e-12)) - np.log(train)


def macro_f1(labels, scores: np.ndarray, average: str = "macro") -> float:
    k = scores.shape[1]
    return classification_metrics(np.asarray(labels), scores.argmax(1), "single_label_classification", average,
                                  [str(i) for i in range(k)], labels_list=list(range(k)))["f1"]


def tune_tau(dev_scores: np.ndarray, dev_labels, shift: np.ndarray, grid, average: str = "macro") -> tuple[float, float]:
    """(tau with the best dev F1, that F1). Ties go to the tau listed first (the grid starts at 0)."""
    best = None
    for tau in grid:
        f1 = macro_f1(dev_labels, dev_scores + tau * shift, average)
        if best is None or f1 > best[1] + 1e-12:
            best = (float(tau), f1)
    return best
