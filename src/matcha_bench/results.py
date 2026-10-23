"""Results tables from finished runs (configs/results.yaml, scripts/make_tables.py).

A run is found under one or more output roots: a local `outputs/`, or the restored copy inside each Kaggle
session's output. Copies of a run must agree; the one with saved test predictions is used, and its score is
recomputed from them. Model pairs are compared with a paired bootstrap that resamples the shared test set:
every model is scored on the same resamples, so each difference is paired item by item.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .finetune.metrics import SINGLE, classification_metrics
from .utils import load_json

PREDICTIONS = "test_predictions.npz"


@dataclass
class Run:
    task: str
    model: str
    seed: int | None
    path: Path  # run directory of the copy used
    metrics: dict
    copies: int  # how many roots hold this run

    @property
    def test(self) -> dict:
        return self.metrics.get("test", self.metrics)  # sanity runs keep the scores at the top level

    @property
    def f1(self) -> float:
        return float(self.test["f1"])

    @property
    def has_predictions(self) -> bool:
        return (self.path / PREDICTIONS).exists()

    def predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Labels and predictions in test-set order (LLM runs store the item indices)."""
        data = np.load(self.path / PREDICTIONS)
        labels, preds = data["labels"], data["preds"]
        if "indices" in data.files:
            order = np.argsort(data["indices"], kind="stable")
            if not np.array_equal(data["indices"][order], np.arange(len(order))):
                raise ValueError(f"{self.path}: predictions do not cover the test set exactly once")
            labels, preds = labels[order], preds[order]
        return labels, preds


def expand_roots(patterns: list[str], base: Path) -> list[Path]:
    roots = []
    for pattern in patterns:
        for p in sorted(glob.glob(str(base / pattern))):
            if Path(p).is_dir() and Path(p) not in roots:
                roots.append(Path(p))
    return roots


def model_seeds(model: dict, task: str, default: list[int]) -> list:
    seeds = model.get("seeds", default)
    return seeds.get(task, default) if isinstance(seeds, dict) else seeds


def _dotted(d: dict, key: str):
    for part in key.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def run_path(model_key: str, model: dict, task: str, seed) -> str:
    tag = model.get("tag_per_task", {}).get(task, model.get("tag", model_key))
    return model["path"].format(task=task, tag=tag, seed=seed)


def find_run(roots: list[Path], model_key: str, model: dict, task: str, seed) -> Run | None:
    rel = run_path(model_key, model, task, seed)
    found = []
    for root in roots:
        path = root / rel / "metrics.json"
        if path.exists():
            m = load_json(path)
            if all(_dotted(m, k) == v for k, v in model.get("match", {}).items()):
                found.append((root / rel, m))
    if not found:
        return None
    if any(not m.get("test_subset") for _, m in found):  # a test on a sample is superseded by the full test
        found = [(d, m) for d, m in found if not m.get("test_subset")]
    scores = {round(float(m.get("test", m)["f1"]), 12) for _, m in found}
    if len(scores) > 1:
        listing = ", ".join(f"{d} = {m.get('test', m)['f1']:.4f}" for d, m in found)
        raise ValueError(f"copies of {rel} disagree ({listing}); add a `match:` rule for {model_key}")
    found.sort(key=lambda x: not (x[0] / PREDICTIONS).exists())  # stable: first root with predictions
    path, metrics = found[0]
    return Run(task, model_key, seed, path, metrics, len(found))


def check_run(run: Run) -> float | None:
    """Recompute the run's primary F1 from its saved predictions; raises if it differs from metrics.json.
    Returns the recomputed value, or None when the run kept no predictions."""
    if not run.has_predictions:
        return None
    labels, preds = run.predictions()
    m = run.metrics
    problem_type = m.get("problem_type", SINGLE)
    label_names = m.get("label_names") or [str(i) for i in range(int(labels.max()) + 1)]
    labels_list = list(range(len(label_names))) if problem_type == SINGLE and (preds < 0).any() else None
    f1 = classification_metrics(labels, preds, problem_type, run.test["average"], label_names, labels_list)["f1"]
    if abs(f1 - run.f1) > 1e-9:
        raise ValueError(f"{run.path}: F1 from predictions {f1:.6f} != metrics.json {run.f1:.6f}")
    return f1


def macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Macro-F1 from confusion counts shaped (..., K, K+1): [true class, predicted class], where the last
    column holds predictions outside the K classes (unparsed LLM answers). Those count as a miss for the
    true class and as nobody's positive, as in `classification_metrics` with an explicit label list."""
    k = counts.shape[-2]
    tp = np.diagonal(counts[..., :k], axis1=-2, axis2=-1)
    fn = counts.sum(-1) - tp
    fp = counts[..., :k].sum(-2) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(denom.shape), where=denom > 0)
    return f1.mean(-1)


def bootstrap_macro_f1(labels: np.ndarray, preds: list[np.ndarray], n_classes: int, n_resamples: int = 10_000,
                       seed: int = 0, chunk: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 of each prediction vector on the full test set and on `n_resamples` resamples of its items
    (drawn with replacement, the same draws for every vector). Returns (observed (R,), resampled (B, R))."""
    n, k, r = len(labels), n_classes, len(preds)
    width = k * (k + 1)
    cols = []
    for i, p in enumerate(preds):
        p = np.where((p >= 0) & (p < k), p, k)
        cols.append(i * width + labels * (k + 1) + p)
    rows = np.tile(np.arange(n), r)
    onehot = sparse.csr_matrix((np.ones(n * r), (rows, np.concatenate(cols))), shape=(n, r * width))
    by_cell = onehot.T.tocsr()
    observed = macro_f1_from_counts(np.asarray(onehot.sum(0)).reshape(r, k, k + 1))
    rng = np.random.default_rng(seed)
    resampled = np.empty((n_resamples, r))
    for start in range(0, n_resamples, chunk):
        b = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, (b, n)) + (np.arange(b) * n)[:, None]
        weights = np.bincount(idx.ravel(), minlength=b * n).reshape(b, n).astype(np.float64)
        counts = np.asarray(by_cell @ weights.T).T.reshape(b, r, k, k + 1)
        resampled[start:start + b] = macro_f1_from_counts(counts)
    return observed, resampled


def compare(observed_a: np.ndarray, boot_a: np.ndarray, observed_b: np.ndarray, boot_b: np.ndarray) -> dict:
    """Difference in mean F1 (A - B) over each model's runs, with a percentile 95% CI and a two-sided
    p-value from the paired resamples. Columns of boot_* are runs (seeds) of that model."""
    d = boot_a.mean(1) - boot_b.mean(1)
    b = len(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).sum() + 1, (d >= 0).sum() + 1) / (b + 1))
    return {"delta": float(observed_a.mean() - observed_b.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(p)}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error), in the input order."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()
