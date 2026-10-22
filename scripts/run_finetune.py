"""Step 6: fine-tune every (task, model, seed) combination. Finished runs are skipped, so the
script can simply be re-launched after a Kaggle session times out.

    python scripts/run_finetune.py --tasks suicide cognitive --models bert-wwm-ext matcha-depression
    python scripts/run_finetune.py --tasks all --models all
"""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.finetune.run import run_finetune


def hparams(cfg: dict, task: str, tag: str) -> dict:
    """defaults < task_overrides[task] < model_overrides[tag] (a model run with its own paper's recipe)."""
    return {**cfg["defaults"], **(cfg.get("task_overrides") or {}).get(task, {}),
            **(cfg.get("model_overrides") or {}).get(tag, {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--tasks", nargs="+", required=True, help="task names, or 'all' for every built task")
    ap.add_argument("--models", nargs="+", required=True, help="model tags from the config, or 'all'")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--set", nargs="*", default=[], help="e.g. defaults.epochs=1")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)

    task_root = resolve_path(cfg["task_root"])
    if args.tasks == ["all"]:
        tasks = sorted(p.name for p in task_root.iterdir() if (p / "meta.json").exists())
    else:
        tasks = args.tasks
    models = list(cfg["models"]) if args.models == ["all"] else args.models
    seeds = args.seeds or cfg["seeds"]

    for task in tasks:
        for tag in models:
            hp = hparams(cfg, task, tag)
            source = str(cfg["models"].get(tag, tag))
            local = resolve_path(source)
            if local.exists():
                model_path = str(local)
            elif source.startswith(("outputs/", "./", "/")):
                print(f"skip {tag}: {local} not found (run DAPT first)")
                continue
            else:
                model_path = source  # Hugging Face hub id
            for seed in seeds:
                run_finetune(task_root, task, model_path, tag, resolve_path(cfg["output_root"]), seed, hp)


if __name__ == "__main__":
    main()
