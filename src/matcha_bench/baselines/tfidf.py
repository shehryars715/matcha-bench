"""Character n-gram TF-IDF + logistic regression: a CPU-only baseline that needs no segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from ..finetune.metrics import MULTI, classification_metrics
from ..tasks import load_task
from ..utils import get_logger, save_json

log = get_logger()
MODEL_TAG = "tfidf-lr"


def run_tfidf(task_root: Path, task: str, output_root: Path, Cs=(0.25, 1.0, 4.0), class_weight: str | None = "balanced") -> dict:
    meta, splits = load_task(task_root, task)
    multi = meta["problem_type"] == MULTI
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=300_000, sublinear_tf=True)
    X = {s: (vec.fit_transform if s == "train" else vec.transform)(df["text"]) for s, df in splits.items()}
    y = {s: np.asarray(df["label"].tolist()) for s, df in splits.items()}

    def make(C):
        lr = LogisticRegression(C=C, max_iter=2000, class_weight=None if multi else class_weight)
        return OneVsRestClassifier(lr) if multi else lr

    def score(model, split):
        preds = model.predict(X[split])
        return classification_metrics(y[split], preds, meta["problem_type"], meta["average"], meta["label_names"]), preds

    best = None
    for C in Cs:
        model = make(C).fit(X["train"], y["train"])
        dev, _ = score(model, "dev")
        log.info("%s | tfidf C=%.2f | dev F1 %.4f", task, C, dev["f1"])
        if best is None or dev["f1"] > best[1]["f1"]:
            best = (C, dev, model)
    C, dev, model = best
    test, preds = score(model, "test")

    out_dir = Path(output_root) / task / MODEL_TAG / "seed0"
    out_dir.mkdir(parents=True, exist_ok=True)
    if multi:
        np.savez_compressed(out_dir / "test_predictions.npz", preds=preds, labels=y["test"])
    else:  # log-probabilities too, on dev as well, as the BERT-type runs save their logits
        for split in ("dev", "test"):
            np.savez_compressed(out_dir / f"{split}_predictions.npz", logits=model.predict_log_proba(X[split]),
                                preds=preds if split == "test" else model.predict(X[split]), labels=y[split])
    record = {
        "task": task, "model_tag": MODEL_TAG, "model_path": "tfidf-char-1-3+logreg", "seed": 0,
        "problem_type": meta["problem_type"], "average": meta["average"], "label_names": meta["label_names"],
        "hyperparameters": {"C": C, "class_weight": class_weight}, "dev": dev, "test": test,
    }
    save_json(record, out_dir / "metrics.json")
    log.info("%s | tfidf-lr | test %s-F1 = %.4f", task, meta["average"], test["f1"])
    return record
