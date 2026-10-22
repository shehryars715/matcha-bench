"""Sanity-check task for the RoBERTa baseline: download Murarka et al.'s Reddit data (their own
train/val/test CSVs, from the Google Drive folder linked in github.com/amurark/mental-health-detection)
and build the task reddit_posts under the task_root of configs/finetune_reddit.yaml.

Runs on Kaggle (Google Drive is not reachable from every network) and needs gdown. Does nothing once
the task exists, so later sessions skip it.

    python scripts/download_reddit_data.py
"""

import argparse
import tempfile
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.tasks import save_task
from matcha_bench.tasks.reddit_tasks import DRIVE_FOLDER_ID, LABELS, build_reddit_posts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/finetune_reddit.yaml", help="the task goes to its task_root")
    ap.add_argument("--raw", default=str(Path(tempfile.gettempdir()) / "murarka-reddit"),
                    help="download folder (the raw CSVs are not kept in outputs/)")
    args = ap.parse_args()
    task_root = resolve_path(load_config(resolve_path(args.config))["task_root"])
    if (task_root / "reddit_posts" / "meta.json").exists():
        print(f"reddit_posts already built under {task_root}")
        return
    raw = Path(args.raw)
    if not any(raw.rglob("posts_train.csv")):
        import gdown

        if not gdown.download_folder(id=DRIVE_FOLDER_ID, output=str(raw), quiet=True):
            raise SystemExit("Google Drive download failed")
        for archive in raw.rglob("*.zip"):
            zipfile.ZipFile(archive).extractall(archive.parent)
    task = build_reddit_posts(raw)
    out = save_task(task, task_root)
    total = 0
    for split, df in task.splits.items():
        counts = df["label"].value_counts().sort_index()
        print(f"{split}: {len(df)} posts", {LABELS[i]: int(n) for i, n in counts.items()})
        total += len(df)
    print(f"built {out}: {total} posts in all (paper: 17,159)")


if __name__ == "__main__":
    main()
