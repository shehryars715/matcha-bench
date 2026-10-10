"""Downstream tasks derived from MATCHA.

All tasks reuse the global user-level split from ``prepare_matcha``, so a user is in the same
split in every task and in the DAPT corpus (train users only).

Post level
    ip_post          Is this Super-Topic post an explicit self-report (diagnosis, medication,
                     hospitalisation)? Positives are IP anchor posts. Negatives are Super-Topic
                     posts from users never identified as positive in that cohort.
    disorder_post    Which of the 7 Super-Topic communities does this post come from?

User level (homepage timeline only; Super-Topic posts are excluded because they leak the label)
    user_binary      Disorder cohort vs clean control.
    user_multiclass  8-way: 7 disorders + control (single-cohort users only).

Every builder accepts ``mask=True`` to replace disorder names and drug names (see ``terms.py``).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ..data.matcha import COHORTS, CONTROL, DISORDERS
from ..terms import mask_shortcut_terms
from .common import SINGLE, TaskData

USER_SEP = " ｜ "


def _cap_per_class(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if not cap:
        return df
    return pd.concat([g.sample(n=min(len(g), cap), random_state=seed) for _, g in df.groupby("label")])


def _cap_total(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    """Stratified down-sampling that keeps the natural label distribution."""
    if not cap or len(df) <= cap:
        return df
    frac = cap / len(df)
    return pd.concat([g.sample(frac=frac, random_state=seed) for _, g in df.groupby("label")])


def _finalise(df: pd.DataFrame, mask: bool, seed: int) -> dict[str, pd.DataFrame]:
    if mask:
        df = df.assign(text=df["text"].map(mask_shortcut_terms))
    df = df[df["text"].str.len() > 0]
    return {s: g.sample(frac=1.0, random_state=seed).reset_index(drop=True) for s, g in df.groupby("split")}


# --------------------------------------------------------------------------------------------
# Post level
# --------------------------------------------------------------------------------------------


def build_ip_post(
    posts, users, ip, *, name: str, mask: bool = False, neg_per_pos_train: int = 3,
    max_eval: int | None = 20000, seed: int = 42, **_,
) -> TaskData:
    stp = posts[posts["source"] == "STP"]
    anchors = set(ip["anchor_note_id"])
    ip_pairs = set(zip(ip["user_id"], ip["cohort"]))
    neg = stp[~stp["note_id"].isin(anchors)]
    neg = neg[[pair not in ip_pairs for pair in zip(neg["user_id"], neg["cohort"])]]
    neg = neg.drop_duplicates("note_id").assign(label=0)
    pos = ip.drop_duplicates("anchor_note_id").assign(label=1)
    pos = pos[pos["text"].str.len() >= 2]
    df = pd.concat([pos[["user_id", "text", "label", "split"]], neg[["user_id", "text", "label", "split"]]])

    parts = []
    for split, g in df.groupby("split"):
        if split == "train":
            n_pos = int((g["label"] == 1).sum())
            negs = g[g["label"] == 0]
            negs = negs.sample(n=min(len(negs), neg_per_pos_train * n_pos), random_state=seed)
            parts.append(pd.concat([g[g["label"] == 1], negs]))
        else:
            parts.append(_cap_total(g, max_eval, seed))
    return TaskData(
        name=name,
        splits=_finalise(pd.concat(parts), mask, seed),
        label_names=["not_self_report", "self_report"],
        average="binary",
        description="Post-level detection of explicit self-reported diagnosis/treatment (IP anchor posts).",
        max_length=128,
    )


def build_disorder_post(
    posts, users, ip, *, name: str, mask: bool = False, max_per_class_train: int | None = 5000,
    max_eval: int | None = 20000, seed: int = 42, **_,
) -> TaskData:
    single = set(users.loc[users["stratum"].isin(DISORDERS), "user_id"])
    stp = posts[(posts["source"] == "STP") & posts["user_id"].isin(single)].drop_duplicates("note_id")
    stp = stp.assign(label=stp["cohort"].map(DISORDERS.index))
    parts = []
    for split, g in stp.groupby("split"):
        parts.append(_cap_per_class(g, max_per_class_train, seed) if split == "train" else _cap_total(g, max_eval, seed))
    return TaskData(
        name=name,
        splits=_finalise(pd.concat(parts)[["user_id", "text", "label", "split"]], mask, seed),
        label_names=list(DISORDERS),
        average="macro",
        description="Post-level 7-way classification of the Super-Topic community a post was written in.",
        max_length=128,
    )


# --------------------------------------------------------------------------------------------
# User level
# --------------------------------------------------------------------------------------------


def _user_documents(posts: pd.DataFrame, user_ids: set, mask: bool, max_chars: int) -> pd.Series:
    hp = posts[(posts["source"] == "HP") & posts["user_id"].isin(user_ids)]
    hp = hp.drop_duplicates(["user_id", "note_id"]).sort_values("created_at", ascending=False)
    if mask:
        hp = hp.assign(text=hp["text"].map(mask_shortcut_terms))

    def join(texts: pd.Series) -> str:
        out, total = [], 0
        for t in texts:
            out.append(t)
            total += len(t) + len(USER_SEP)
            if total >= max_chars:
                break
        return USER_SEP.join(out)[:max_chars]

    return hp.groupby("user_id")["text"].agg(join)


def _eligible_users(posts: pd.DataFrame, users: pd.DataFrame, min_posts: int) -> pd.DataFrame:
    hp_counts = posts[posts["source"] == "HP"].drop_duplicates(["user_id", "note_id"]).groupby("user_id").size()
    users = users.assign(n_hp_clean=users["user_id"].map(hp_counts).fillna(0).astype(int))
    return users[users["n_hp_clean"] >= min_posts]


def build_user_binary(
    posts, users, ip, *, name: str, mask: bool = False, positives: str = "all", min_posts: int = 5,
    max_chars: int = 600, pos_per_neg: float = 1.0, seed: int = 42, **_,
) -> TaskData:
    pool = _eligible_users(posts, users, min_posts)
    neg = pool[pool["stratum"] == CONTROL].assign(label=0)
    pos = pool[pool["stratum"] != CONTROL]
    if positives == "ip":
        pos = pos[pos["ip_cohorts"].map(len) > 0]
    pos = pos.assign(label=1)

    parts = []
    for split in ("train", "dev", "test"):
        n = int((neg["split"] == split).sum())
        p = pos[pos["split"] == split]
        p = p.sample(n=min(len(p), int(round(pos_per_neg * n))), random_state=seed)
        parts.append(pd.concat([neg[neg["split"] == split], p]))
    df = pd.concat(parts)[["user_id", "label", "split"]]
    docs = _user_documents(posts, set(df["user_id"]), mask, max_chars)
    df = df.assign(text=df["user_id"].map(docs).fillna(""))
    return TaskData(
        name=name,
        splits=_finalise(df, False, seed),
        label_names=["control", "disorder"],
        average="macro",
        description=f"User-level disorder ({positives} users) vs control from homepage posts only.",
        max_length=512,
        extra={"min_posts": min_posts, "max_chars": max_chars, "positives": positives, "masked": mask},
    )


def build_user_multiclass(
    posts, users, ip, *, name: str, mask: bool = False, min_posts: int = 5, max_chars: int = 600,
    max_per_class_train: int | None = 1500, max_per_class_eval: int | None = 500, seed: int = 42, **_,
) -> TaskData:
    pool = _eligible_users(posts, users, min_posts)
    pool = pool[pool["stratum"].isin(COHORTS)]
    pool = pool.assign(label=pool["stratum"].map(COHORTS.index))
    parts = []
    for split, g in pool.groupby("split"):
        parts.append(_cap_per_class(g, max_per_class_train if split == "train" else max_per_class_eval, seed))
    df = pd.concat(parts)[["user_id", "label", "split"]]
    docs = _user_documents(posts, set(df["user_id"]), mask, max_chars)
    df = df.assign(text=df["user_id"].map(docs).fillna(""))
    return TaskData(
        name=name,
        splits=_finalise(df, False, seed),
        label_names=list(COHORTS),
        average="macro",
        description="User-level 8-way classification (7 disorders + control) from homepage posts only.",
        max_length=512,
        extra={"min_posts": min_posts, "max_chars": max_chars, "masked": mask},
    )


BUILDERS: dict[str, Callable[..., TaskData]] = {
    "ip_post": build_ip_post,
    "disorder_post": build_disorder_post,
    "user_binary": build_user_binary,
    "user_multiclass": build_user_multiclass,
}
