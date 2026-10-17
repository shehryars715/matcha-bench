"""Data for the LLM baseline: MentalGLM's own test sets (for the sanity check) and fixed subsets.

The sanity check uses the test files from the MentalGLM repository, so the scores compare
directly with the paper (SOS-HL-1K F1 85.12, SocialCD-3K micro-F1 71.04 for MentalGLM-chat).
Its SocialCD-3K test set is identical to data/tasks/cognitive/test (682 posts, same label
order); 240 of its 250 SOS-HL-1K test posts are in ours.

A 9B model is too slow to train on every MATCHA training example or to label all 20,000 test
posts of the post-level tasks on a T4, so those use fixed subsets. `subset_indices` is
deterministic, so the BERT baselines can be re-scored on the same test items from their saved
predictions (test_predictions.npz keeps the test-file order).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MENTALGLM_RAW = "https://raw.githubusercontent.com/zwzzzQAQ/MentalGLM/main/Raw%20data/"
MENTALGLM_TEST = {"suicide": "Suiside/test_set.tsv", "cognitive": "CD/test_data.tsv"}


def load_mentalglm_test(task: str, cache_dir: Path) -> pd.DataFrame:
    """MentalGLM's test file for `task` as a frame with text and label (int, or a 0/1 list)."""
    path = Path(cache_dir) / MENTALGLM_TEST[task].replace("/", "_")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MENTALGLM_RAW + MENTALGLM_TEST[task], path)
    df = pd.read_csv(path, sep="\t", header=None, quoting=3, dtype=str, keep_default_na=False)
    if task == "suicide":  # label, text
        return pd.DataFrame({"text": df[1], "label": df[0].astype(int)})
    labels = df.iloc[:, :12].astype(int).values.tolist()  # 12 distortion columns, then the text
    return pd.DataFrame({"text": df[12], "label": labels})


def subset_indices(n: int, k: int | None, seed: int = 0, labels=None) -> np.ndarray:
    """Sorted positions of a fixed random subset of size k (all of them if k is None or >= n).

    With `labels`, the subset is stratified: every class keeps its share (largest-remainder
    rounding, at least one item per class), and the size is exactly k.
    """
    if k is None or k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(n, size=k, replace=False))
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if k < len(classes):
        raise ValueError(f"subset of {k} cannot hold all {len(classes)} classes")
    share = counts / n * k
    quota = np.maximum(1, np.floor(share).astype(int))
    for i in np.argsort(-(share - np.floor(share)), kind="stable"):  # hand out the remaining places
        if quota.sum() >= k:
            break
        if quota[i] < counts[i]:
            quota[i] += 1
    while quota.sum() > k:  # only if the at-least-one rule overshot: trim the largest classes
        quota[np.argmax(quota)] -= 1
    picked = [rng.choice(np.flatnonzero(labels == c), size=q, replace=False) for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(picked))
