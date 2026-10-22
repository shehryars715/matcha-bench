"""Sanity-check data for the RoBERTa baseline: Murarka, Radhakrishnan & Ravichandran (LOUHI 2021),
"Classification of mental illnesses on social media using RoBERTa".

17,159 English Reddit posts from 13 subreddits in six classes (5 disorders + none), with the
authors' own train/val/test files (Google Drive folder linked from
https://github.com/amurark/mental-health-detection). Only post bodies are used: the paper's
"posts" setting (RoBERTa F1 0.86); MATCHA posts have no titles. The paper's overall scores equal
the mean of its per-class scores (Table 3 vs Table 4), so the task is macro-averaged like MATCHA.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import TaskData

DRIVE_FOLDER_ID = "11aW_fpXjA-O51uv3xYY3xj6NWGh1VYh_"
LABELS = ["adhd", "anxiety", "bipolar", "depression", "ptsd", "none"]  # the files' class_id 0-5
SPLIT_FILES = {"train": "posts_train.csv", "dev": "posts_val.csv", "test": "posts_test.csv"}


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    names = df["class_name"].astype(str).str.strip().str.lower()
    unknown = sorted(set(names) - set(LABELS))
    if unknown:
        raise ValueError(f"{path.name}: unknown classes {unknown}")
    labels = names.map(LABELS.index)
    if "class_id" in df.columns and not (df["class_id"].astype(int) == labels).all():
        raise ValueError(f"{path.name}: class_id does not follow the README's order {LABELS}")
    ids = df["id"].astype(str) if "id" in df.columns else [f"{path.stem}-{i}" for i in range(len(df))]
    return pd.DataFrame({"id": ids, "text": df["post"].fillna("").astype(str), "label": labels})


def build_reddit_posts(raw: Path) -> TaskData:
    splits = {}
    for split, name in SPLIT_FILES.items():
        found = sorted(Path(raw).rglob(name))
        if not found:
            raise FileNotFoundError(f"{name} not found under {raw}")
        splits[split] = _read(found[0])
    return TaskData(
        name="reddit_posts", group="reddit", splits=splits, label_names=LABELS, average="macro",
        max_length=512, description="Murarka et al. (2021) Reddit posts: 5 disorders + none (sanity check).",
    )
