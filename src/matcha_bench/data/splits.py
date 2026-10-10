"""User-level, stratified train/dev/test splits.

Splitting by user (never by post) is the main leakage control in the benchmark: a person's
posts never appear on both sides of a split, and the DAPT corpus is built from
training-split users only, so downstream test users are never seen during pre-training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("train", "dev", "test")


def assign_user_splits(
    users: pd.DataFrame,
    strata_col: str = "stratum",
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> pd.Series:
    """Return a Series (index aligned with ``users``) with values in {train, dev, test}."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    train_r, dev_r, test_r = ratios
    strata = users[strata_col].to_numpy(dtype=object)
    labels = np.full(len(users), "train", dtype=object)

    values, counts = np.unique(strata, return_counts=True)
    # Strata too small to stratify go to train; they are tiny and only affect DAPT data.
    rare = np.isin(strata, values[counts < 10])
    eligible = np.flatnonzero(~rare)
    if len(eligible):
        rest, test = train_test_split(eligible, test_size=test_r, stratify=strata[eligible], random_state=seed)
        dev_share = dev_r / (train_r + dev_r)
        _, dev = train_test_split(rest, test_size=dev_share, stratify=strata[rest], random_state=seed)
        labels[dev] = "dev"
        labels[test] = "test"
    return pd.Series(labels, index=users.index, dtype=object)
