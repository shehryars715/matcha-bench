"""Loading and preparing the raw MATCHA release.

Raw layout (one CSV per cohort)::

    Homepage_Posts/homepage_posts_{COHORT}.csv        user_id, gender, content, note_id, create_date_time
    Super_Topic_Posts/super_topic_posts_{DISORDER}.csv same columns (no CONTROL file)
    Identified_Positive_User/positive_users_{DISORDER}.csv  user_id, anchor_note, anchor_note_id

``prepare_matcha`` writes three tables to ``data/processed``:

* ``posts.parquet``  one row per (cohort, source, note_id), with cleaned text
* ``users.parquet``  one row per user: cohorts, IP cohorts, gender, stratum, split
* ``ip.parquet``     IP anchor posts (cleaned)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..text import clean_text, is_repost, is_truncated
from ..utils import get_logger, save_json
from .splits import assign_user_splits

DISORDERS = ("ADHD", "ASD", "BD", "DEP", "GAD", "OCD", "SZ")
CONTROL = "CONTROL"
COHORTS = DISORDERS + (CONTROL,)
DISORDER_NAMES = {
    "ADHD": ("attention-deficit/hyperactivity disorder", "多动症"),
    "ASD": ("autism spectrum disorder", "孤独症/自闭症"),
    "BD": ("bipolar disorder", "双相情感障碍"),
    "DEP": ("depression", "抑郁症"),
    "GAD": ("generalized anxiety disorder", "焦虑症"),
    "OCD": ("obsessive-compulsive disorder", "强迫症"),
    "SZ": ("schizophrenia", "精神分裂症"),
}

_SOURCES = {
    "HP": ("Homepage_Posts", "homepage_posts"),
    "STP": ("Super_Topic_Posts", "super_topic_posts"),
}

log = get_logger()


def read_posts(root: Path, cohort: str, source: str) -> pd.DataFrame:
    folder, prefix = _SOURCES[source]
    path = Path(root) / folder / f"{prefix}_{cohort}.csv"
    if not path.exists():
        if not (cohort == CONTROL and source == "STP"):
            log.warning("Missing file %s (skipped)", path)
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["cohort"] = cohort
    df["source"] = source
    return df


def load_raw_posts(root: Path) -> pd.DataFrame:
    frames = [read_posts(root, c, s) for c in COHORTS for s in _SOURCES]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError(f"No MATCHA CSVs found under {root}")
    return pd.concat(frames, ignore_index=True)


def load_raw_ip(root: Path) -> pd.DataFrame:
    frames = []
    for d in DISORDERS:
        path = Path(root) / "Identified_Positive_User" / f"positive_users_{d}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            df["cohort"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["user_id", "anchor_note", "anchor_note_id", "cohort"]
    )


def _mode(series: pd.Series) -> str:
    vc = series[series != ""].value_counts()
    return vc.index[0] if len(vc) else "unknown"


def prepare_matcha(
    root: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    min_chars: int = 2,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"seed": seed, "ratios": list(ratios), "min_chars": min_chars}

    log.info("Reading raw MATCHA CSVs from %s", root)
    posts = load_raw_posts(root)
    stats["raw_rows"] = len(posts)

    posts["is_repost"] = posts["content"].map(is_repost)
    posts["is_truncated"] = posts["content"].map(is_truncated)
    log.info("Cleaning %d posts", len(posts))
    posts["text"] = posts["content"].map(clean_text)
    posts["created_at"] = pd.to_datetime(posts["create_date_time"], errors="coerce", utc=True)

    posts = posts[posts["text"].str.len() >= min_chars]
    stats["after_min_chars"] = len(posts)
    posts = posts.drop_duplicates(["cohort", "source", "note_id"])
    stats["after_note_dedup"] = len(posts)
    posts = posts.drop_duplicates(["user_id", "cohort", "source", "text"])
    stats["after_text_dedup"] = len(posts)

    # ---- users -------------------------------------------------------------------------
    cohorts_by_user = posts.groupby("user_id")["cohort"].agg(lambda s: sorted(set(s)))
    users = cohorts_by_user.rename("cohorts").to_frame()
    users["disorders"] = users["cohorts"].map(lambda cs: [c for c in cs if c != CONTROL])

    # Control users who also appear in a disorder cohort are not clean controls: they
    # keep their disorder membership and lose their control posts.
    contaminated = users.index[users["cohorts"].map(lambda cs: CONTROL in cs and len(cs) > 1)]
    stats["contaminated_control_users"] = len(contaminated)
    drop = (posts["cohort"] == CONTROL) & posts["user_id"].isin(contaminated)
    posts = posts[~drop]
    users.loc[contaminated, "cohorts"] = users.loc[contaminated, "disorders"]

    ip = load_raw_ip(root)
    ip["text"] = ip["anchor_note"].map(clean_text)
    ip_by_user = ip.groupby("user_id")["cohort"].agg(lambda s: sorted(set(s)))
    users["ip_cohorts"] = users.index.map(lambda u: ip_by_user.get(u, []))
    users["is_control"] = users["cohorts"].map(lambda cs: cs == [CONTROL])
    users["gender"] = posts.groupby("user_id")["gender"].agg(_mode)
    counts = posts.drop_duplicates(["user_id", "source", "note_id"]).groupby(["user_id", "source"]).size()
    counts = counts.unstack(fill_value=0)
    users["n_hp"] = counts.get("HP", 0)
    users["n_stp"] = counts.get("STP", 0)
    users[["n_hp", "n_stp"]] = users[["n_hp", "n_stp"]].fillna(0).astype(int)

    def stratum(row) -> str:
        if row["is_control"]:
            return CONTROL
        return row["disorders"][0] if len(row["disorders"]) == 1 else "MULTI"

    users["stratum"] = users.apply(stratum, axis=1)
    users["split"] = assign_user_splits(users, "stratum", ratios, seed)
    users = users.reset_index()

    posts = posts.merge(users[["user_id", "split"]], on="user_id", how="left")
    ip = ip.merge(users[["user_id", "split"]], on="user_id", how="left")
    ip["split"] = ip["split"].fillna("train")

    # ---- stats -------------------------------------------------------------------------
    stats["posts"] = len(posts)
    stats["unique_notes"] = int(posts["note_id"].nunique())
    stats["users"] = len(users)
    stats["users_by_stratum"] = users["stratum"].value_counts().to_dict()
    stats["users_by_split"] = users["split"].value_counts().to_dict()
    stats["multi_disorder_users"] = int((users["stratum"] == "MULTI").sum())
    stats["ip_users"] = int(users["ip_cohorts"].map(len).gt(0).sum())
    stats["posts_by_cohort_source"] = {
        f"{c}/{s}": int(n) for (c, s), n in posts.groupby(["cohort", "source"]).size().items()
    }
    stats["repost_share"] = float(posts["is_repost"].mean())
    stats["truncated_share"] = float(posts["is_truncated"].mean())
    stats["split_by_stratum"] = {
        k: v for k, v in users.groupby(["stratum", "split"]).size().unstack(fill_value=0).to_dict("index").items()
    }

    keep = ["user_id", "cohort", "source", "note_id", "gender", "created_at", "text", "is_repost", "is_truncated", "split"]
    posts[keep].to_parquet(out_dir / "posts.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    ip[["user_id", "cohort", "anchor_note_id", "text", "split"]].to_parquet(out_dir / "ip.parquet", index=False)
    save_json(stats, out_dir / "stats.json")
    log.info("Wrote processed MATCHA to %s (%d posts, %d users)", out_dir, len(posts), len(users))
    return stats


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    missing = [n for n in ("posts", "users", "ip") if not (processed_dir / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {processed_dir}. Run `python scripts/prepare_matcha.py` first."
        )
    posts = pd.read_parquet(processed_dir / "posts.parquet")
    users = pd.read_parquet(processed_dir / "users.parquet")
    ip = pd.read_parquet(processed_dir / "ip.parquet")
    for col in ("cohorts", "disorders", "ip_cohorts"):
        users[col] = users[col].map(list)
    return posts, users, ip
