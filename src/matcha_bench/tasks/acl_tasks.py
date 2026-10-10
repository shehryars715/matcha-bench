"""The public downstream datasets used by Chinese MentalBERT (Zhai et al., Findings of ACL 2024).

| Task            | Source                                   | Split used here                      | Avg.   |
|-----------------|------------------------------------------|--------------------------------------|--------|
| ewect_usual     | SMP2020-EWECT (BrownSweater mirror)      | official train / eval / test          | macro  |
| ewect_epidemic  | SMP2020-EWECT "virus" subset             | official train / eval / test          | macro  |
| waimai          | ChineseNlpCorpus waimai_10k              | 80/10/10 stratified (seed 42)         | binary |
| cognitive       | SocialCD-3K (Qi et al. 2023)             | official test; train -> 80/20 dev     | micro  |
| suicide         | SOS-HL-1K (Qi et al. 2023)               | official val = test; train -> 80/20   | binary |
| c2d2            | C2D2 (Wang et al. 2023), access by request | user-supplied train/dev/test CSVs   | macro  |

The paper's sizes are 27,768/2,000/5,000 (EWECT-usual), 8,606/2,000/3,000 (epidemic),
9,589/1,199/1,199 (Waimai), 2,180/545/682 (Cognitive), 800/199/250 (Suicide).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import MULTI, SINGLE, TaskData

RAW_URLS = {
    "ewect/usual_train.txt": "https://raw.githubusercontent.com/BrownSweater/BERT_SMP2020-EWECT/master/data/clean/usual_train.txt",
    "ewect/usual_eval_labeled.txt": "https://raw.githubusercontent.com/BrownSweater/BERT_SMP2020-EWECT/master/data/clean/usual_eval_labeled.txt",
    "ewect/usual_test_labeled.txt": "https://raw.githubusercontent.com/BrownSweater/BERT_SMP2020-EWECT/master/data/clean/usual_test_labeled.txt",
    "ewect/virus_train.txt": "https://raw.githubusercontent.com/BrownSweater/BERT_SMP2020-EWECT/master/data/clean/virus_train.txt",
    "ewect/virus_eval_labeled.txt": "https://raw.githubusercontent.com/BrownSweater/BERT_SMP2020-EWECT/master/data/clean/virus_eval_labeled.txt",
    "ewect/virus_test_labeled.txt": "https://raw.githubusercontent.com/BrownSweater/BERT_SMP2020-EWECT/master/data/clean/virus_test_labeled.txt",
    "waimai/waimai_10k.csv": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
    "socialcd3k/train_data_with_header.tsv": "https://raw.githubusercontent.com/HongzhiQ/SupervisedVsLLM-EfficacyEval/main/data/SocialCD-3k/train_data_with_header.tsv",
    "socialcd3k/test_data_with_header.tsv": "https://raw.githubusercontent.com/HongzhiQ/SupervisedVsLLM-EfficacyEval/main/data/SocialCD-3k/test_data_with_header.tsv",
    "suicide/suicide_train_BERT.tsv": "https://raw.githubusercontent.com/HongzhiQ/SupervisedVsLLM-EfficacyEval/main/data/suicide/suicide_train_BERT.tsv",
    "suicide/suicide_val_BERT.tsv": "https://raw.githubusercontent.com/HongzhiQ/SupervisedVsLLM-EfficacyEval/main/data/suicide/suicide_val_BERT.tsv",
}

EWECT_LABELS = ["neutral", "happy", "angry", "sad", "fear", "surprise"]
COGNITIVE_LABELS = [
    "all_or_nothing", "overgeneralization", "mental_filter", "disqualifying_positive", "mind_reading",
    "fortune_telling", "magnification", "emotional_reasoning", "should_statements", "labeling",
    "blaming_oneself", "blaming_others",
]


def _split_off_dev(df: pd.DataFrame, dev_size: float, seed: int, stratify: bool = True):
    strat = df["label"] if stratify else None
    train, dev = train_test_split(df, test_size=dev_size, random_state=seed, stratify=strat)
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def _ewect(raw: Path, prefix: str, name: str) -> TaskData:
    def read(fname: str) -> pd.DataFrame:
        with open(raw / "ewect" / fname, encoding="utf-8") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
        return pd.DataFrame({"text": df["content"].astype(str), "label": df["label"].map(EWECT_LABELS.index)})

    return TaskData(
        name=name,
        group="acl",
        splits={
            "train": read(f"{prefix}_train.txt"),
            "dev": read(f"{prefix}_eval_labeled.txt"),
            "test": read(f"{prefix}_test_labeled.txt"),
        },
        label_names=EWECT_LABELS,
        average="macro",
        description=f"SMP2020-EWECT {prefix} six-way emotion classification.",
    )


def build_ewect_usual(raw: Path, seed: int = 42) -> TaskData:
    return _ewect(raw, "usual", "ewect_usual")


def build_ewect_epidemic(raw: Path, seed: int = 42) -> TaskData:
    return _ewect(raw, "virus", "ewect_epidemic")


def build_waimai(raw: Path, seed: int = 42) -> TaskData:
    df = pd.read_csv(raw / "waimai" / "waimai_10k.csv")
    df = pd.DataFrame({"text": df["review"].astype(str), "label": df["label"].astype(int)})
    train, rest = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    dev, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["label"])
    return TaskData(
        name="waimai", group="acl",
        splits={"train": train, "dev": dev, "test": test},
        label_names=["negative", "positive"], average="binary",
        description="Waimai_10k food-delivery review sentiment.",
    )


def build_cognitive(raw: Path, seed: int = 42) -> TaskData:
    def read(fname: str) -> pd.DataFrame:
        df = pd.read_csv(raw / "socialcd3k" / fname, sep="\t", encoding="utf-8")
        label_cols, text_col = df.columns[:12], df.columns[12]
        return pd.DataFrame({
            "text": df[text_col].astype(str),
            "label": df[label_cols].astype(int).values.tolist(),
        })

    train, dev = _split_off_dev(read("train_data_with_header.tsv"), 0.2, seed, stratify=False)
    return TaskData(
        name="cognitive", group="acl",
        splits={"train": train, "dev": dev, "test": read("test_data_with_header.tsv")},
        label_names=COGNITIVE_LABELS, problem_type=MULTI, average="micro",
        description="SocialCD-3K multi-label cognitive distortion detection (12 Burns categories).",
    )


def build_suicide(raw: Path, seed: int = 42) -> TaskData:
    # Two complementary one-hot columns + text. Column 0 == 1 marks HIGH suicide risk
    # (checked against the paired suicide_train.jsonl, where "0 1" rows are 低自杀风险).
    def read(fname: str) -> pd.DataFrame:
        df = pd.read_csv(raw / "suicide" / fname, sep="\t", header=None, encoding="gb18030", encoding_errors="replace")
        return pd.DataFrame({"text": df[2].astype(str), "label": df[0].astype(int)})

    train, dev = _split_off_dev(read("suicide_train_BERT.tsv"), 0.2, seed)
    return TaskData(
        name="suicide", group="acl",
        splits={"train": train, "dev": dev, "test": read("suicide_val_BERT.tsv")},
        label_names=["low_risk", "high_risk"], average="binary",
        description="SOS-HL-1K high vs low suicide risk on tree-hole posts.",
    )


def build_c2d2(raw: Path, seed: int = 42) -> TaskData:
    base = raw / "c2d2"
    splits = {s: pd.read_csv(base / f"{s}.csv") for s in ("train", "dev", "test")}
    names = sorted(set(splits["train"]["label"].astype(str)))
    for s, df in splits.items():
        splits[s] = pd.DataFrame({"text": df["text"].astype(str), "label": df["label"].astype(str).map(names.index)})
    return TaskData(
        name="c2d2", group="acl", splits=splits, label_names=names, average="macro",
        description="C2D2 multi-class cognitive distortion (data by request from its authors).",
    )


BUILDERS = {
    "ewect_usual": build_ewect_usual,
    "ewect_epidemic": build_ewect_epidemic,
    "waimai": build_waimai,
    "cognitive": build_cognitive,
    "suicide": build_suicide,
    "c2d2": build_c2d2,
}
