from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEPRESSION_LEXICON_URL = (
    "https://raw.githubusercontent.com/omfoggynight/Chinese-Depression-domain-Lexicon/master/"
    "Depression%20Domain%20Lexicon_202003.csv"
)
_HANZI_ONLY = re.compile(r"^[一-鿿]+$")


def load_lexicon(path: str | Path) -> set[str]:
    """One word per line; blank lines and lines starting with '#' are ignored."""
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                words.add(w)
    return words


def save_lexicon(words: Iterable[str], path: str | Path, header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            for line in header.strip().splitlines():
                f.write(f"# {line}\n")
        for w in sorted(set(words)):
            f.write(w + "\n")


def apply_word_annotations(candidates: pd.DataFrame, annotations: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join word-level keep decisions (``word, category``: every listed word is kept) onto
    the per-disorder candidates. Returns the annotated candidates and any annotated words
    that are not candidates (typos, or words from another mining run)."""
    ann = annotations.assign(word=annotations["word"].astype(str).str.strip()).drop_duplicates("word")
    out = candidates.merge(ann[["word", "category"]], on="word", how="left")
    out["keep"] = out["category"].notna().astype(int)
    out["category"] = out["category"].fillna("none")
    unmatched = sorted(set(ann["word"]) - set(candidates["word"]))
    return out, unmatched


def convert_depression_lexicon(csv_path: str | Path) -> set[str]:
    """Reproduce the filtering in the Chinese MentalBERT code: keep rows with
    抑郁倾向 == 1 whose word is made only of Chinese characters."""
    df = pd.read_csv(csv_path, encoding="gbk")
    label_col = next(c for c in df.columns if c.startswith("抑郁倾向"))
    word_col = "抑郁词" if "抑郁词" in df.columns else df.columns[0]
    words = df.loc[df[label_col] == 1, word_col].astype(str)
    return {w for w in words if _HANZI_ONLY.match(w)}
