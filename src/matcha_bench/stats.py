"""Significance testing and aggregation over seeds.

The Chinese MentalBERT paper reports single runs with no variance. Here every number is mean ± std over seeds,
and model pairs are compared with a paired bootstrap on the shared test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .finetune.metrics import classification_metrics
from .utils import load_json


def paired_bootstrap(
    labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, metric_fn, n_resamples: int = 1000, seed: int = 0
) -> dict:
    """One-sided paired bootstrap: is A better than B? Returns the observed delta and p-value."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    observed = metric_fn(labels, preds_a) - metric_fn(labels, preds_b)
    worse = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if metric_fn(labels[idx], preds_a[idx]) - metric_fn(labels[idx], preds_b[idx]) <= 0:
            worse += 1
    return {"delta": float(observed), "p_value": (worse + 1) / (n_resamples + 1)}


def collect_results(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in Path(output_root).glob("*/*/seed*/metrics.json"):
        m = load_json(path)
        test = m["test"]
        rows.append({
            "task": m["task"], "model": m["model_tag"], "seed": m["seed"],
            "f1": test["f1"], "macro_f1": test["macro_f1"], "precision": test["precision"],
            "recall": test["recall"], "accuracy": test["accuracy"], "average": test["average"],
        })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Task x model table of 'mean ± std (n)' in percentage points."""
    if results.empty:
        return results
    g = results.groupby(["task", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    g["cell"] = [f"{100 * m:.2f} ± {100 * s:.2f}" for m, s in zip(g["mean"], g["std"])]
    return g.pivot(index="model", columns="task", values="cell")


def load_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(run_dir) / "test_predictions.npz")
    return data["labels"], data["preds"]


def compare_models(output_root: Path, task: str, model_a: str, model_b: str, n_resamples: int = 1000) -> pd.DataFrame:
    root = Path(output_root) / task
    rows = []
    for run_a in sorted((root / model_a).glob("seed*")):
        run_b = root / model_b / run_a.name
        if not (run_b / "test_predictions.npz").exists():
            continue
        meta = load_json(run_a / "metrics.json")
        labels, pa = load_predictions(run_a)
        _, pb = load_predictions(run_b)

        def metric_fn(y, p, meta=meta):
            return classification_metrics(y, p, meta["problem_type"], meta["average"], meta["label_names"])["f1"]

        res = paired_bootstrap(labels, pa, pb, metric_fn, n_resamples=n_resamples)
        rows.append({"task": task, "seed": run_a.name, "a": model_a, "b": model_b, **res})
    return pd.DataFrame(rows)
