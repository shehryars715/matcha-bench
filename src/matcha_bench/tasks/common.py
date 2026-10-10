"""A single on-disk format for every downstream task (MATCHA and ACL alike).

    data/tasks/<name>/
        meta.json        problem_type, label_names, average, ...
        train.jsonl      {"id": ..., "text": ..., "label": int | list[int]}
        dev.jsonl
        test.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..utils import load_json, save_json

SINGLE = "single_label_classification"
MULTI = "multi_label_classification"
SPLITS = ("train", "dev", "test")


@dataclass
class TaskData:
    name: str
    splits: dict[str, pd.DataFrame]
    label_names: list[str]
    problem_type: str = SINGLE
    average: str = "macro"  # primary F1 averaging: macro | binary | micro
    group: str = "matcha"
    description: str = ""
    max_length: int = 128
    extra: dict = field(default_factory=dict)

    @property
    def num_labels(self) -> int:
        return len(self.label_names)


def _label_distribution(df: pd.DataFrame, task: TaskData) -> dict:
    if task.problem_type == MULTI:
        totals = [0] * task.num_labels
        for labels in df["label"]:
            for i, v in enumerate(labels):
                totals[i] += int(v)
        return dict(zip(task.label_names, totals))
    counts = df["label"].value_counts().to_dict()
    return {task.label_names[int(k)]: int(v) for k, v in sorted(counts.items())}


def save_task(task: TaskData, root: Path) -> Path:
    out = Path(root) / task.name
    out.mkdir(parents=True, exist_ok=True)
    sizes, dists = {}, {}
    for split in SPLITS:
        df = task.splits[split].reset_index(drop=True)
        if "id" not in df.columns:
            df.insert(0, "id", [f"{split}-{i}" for i in range(len(df))])
        df[["id", "text", "label"]].to_json(out / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)
        sizes[split] = len(df)
        dists[split] = _label_distribution(df, task)
    save_json(
        {
            "name": task.name,
            "group": task.group,
            "description": task.description,
            "problem_type": task.problem_type,
            "label_names": task.label_names,
            "average": task.average,
            "max_length": task.max_length,
            "sizes": sizes,
            "label_distribution": dists,
            **task.extra,
        },
        out / "meta.json",
    )
    return out


def load_task(root: Path, name: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    task_dir = Path(root) / name
    if not (task_dir / "meta.json").exists():
        raise FileNotFoundError(f"Task {name!r} not built under {root}. Run scripts/build_tasks.py first.")
    meta = load_json(task_dir / "meta.json")
    splits = {
        s: pd.read_json(task_dir / f"{s}.jsonl", orient="records", lines=True, dtype={"id": str, "text": str})
        for s in SPLITS
    }
    return meta, splits
