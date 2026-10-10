"""Agreement between two lexicon annotators, in place of a human spot-check.

Annotator A gives word-level decisions (``word,category``: listed words are kept).
Annotator B gives (disorder, word) decisions (e.g. DeepSeek majority votes).

Two views are reported:
* pair level: every mined (disorder, word) candidate, A's decision copied across disorders;
* word level: B keeps a word if it kept it for any disorder (the lexicon is a union, so this
  matches how both lexicons are actually built).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _summary(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.astype(int), b.astype(int)
    return {
        "n": int(len(a)),
        "percent_agreement": float((a == b).mean()),
        "cohen_kappa": float(cohen_kappa_score(a, b)),
        "both_keep": int(((a == 1) & (b == 1)).sum()),
        "only_a_keeps": int(((a == 1) & (b == 0)).sum()),
        "only_b_keeps": int(((a == 0) & (b == 1)).sum()),
        "both_reject": int(((a == 0) & (b == 0)).sum()),
        "a_keep_rate": float(a.mean()),
        "b_keep_rate": float(b.mean()),
    }


def annotation_agreement(candidates: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    a_cat = dict(zip(ann_a["word"].astype(str).str.strip(), ann_a["category"]))
    pairs = candidates[["disorder", "word"]].merge(
        ann_b[["disorder", "word", "keep", "category"]].rename(columns={"keep": "b_keep", "category": "b_category"}),
        on=["disorder", "word"], how="left",
    )
    pairs["b_keep"] = pairs["b_keep"].fillna(0).astype(int)
    pairs["a_keep"] = pairs["word"].map(lambda w: int(w in a_cat))
    pairs["a_category"] = pairs["word"].map(lambda w: a_cat.get(w, "none"))

    words = pairs.groupby("word").agg(
        a_keep=("a_keep", "first"),
        a_category=("a_category", "first"),
        b_keep=("b_keep", "max"),
        b_keep_share=("b_keep", "mean"),
        b_category=("b_category", lambda s: Counter(c for c in s if c not in ("none", "missing")).most_common(1)[0][0]
                    if any(c not in ("none", "missing") for c in s) else "none"),
        disorders=("disorder", lambda s: ",".join(sorted(s))),
    ).reset_index()

    both = words[(words["a_keep"] == 1) & (words["b_keep"] == 1)]
    report = {
        "pair_level": _summary(pairs["a_keep"], pairs["b_keep"]),
        "word_level": _summary(words["a_keep"], words["b_keep"]),
        "category_agreement_on_shared_keeps": float((both["a_category"] == both["b_category"]).mean()) if len(both) else None,
    }
    disagreements = words[words["a_keep"] != words["b_keep"]].sort_values(["a_keep", "b_keep_share"], ascending=False)
    return report, disagreements
