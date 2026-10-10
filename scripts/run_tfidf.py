"""CPU baseline: char n-gram TF-IDF + logistic regression on every built task."""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.baselines import run_tfidf
from matcha_bench.config import load_config, resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--tasks", nargs="+", default=["all"])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    task_root = resolve_path(cfg["task_root"])
    if args.tasks == ["all"]:
        tasks = sorted(p.name for p in task_root.iterdir() if (p / "meta.json").exists())
    else:
        tasks = args.tasks
    for task in tasks:
        run_tfidf(task_root, task, resolve_path(cfg["output_root"]))


if __name__ == "__main__":
    main()
