"""LLM baseline: LoRA fine-tuning of MentalGLM on the MATCHA tasks (one run per task and seed).

    python scripts/run_llm_finetune.py --tasks matcha_disorder_post_masked --models mentalglm-chat
    python scripts/run_llm_finetune.py --tasks all --models mentalglm-chat --seeds 42
    python scripts/run_llm_finetune.py --tasks matcha_user_binary_masked --models mentalglm-chat --probe-steps 6

Finished runs (metrics.json present) are skipped. `--probe-steps N` only times N optimizer steps
and a small generation pass, writing outputs/llm/finetune/_probe/<task>__<model>.json, to plan
GPU time. Same command-line shape as scripts/run_finetune.py, so the pipeline can run either.
"""

import argparse
import os

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.llm.convert import ensure_converted
from matcha_bench.llm.train import run_llm_finetune


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--probe-steps", type=int)
    ap.add_argument("--prepare-only", action="store_true", help="download and convert the models, then stop")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. defaults.epochs=1")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    task_root = resolve_path(cfg["task_root"])
    tasks = sorted(cfg["task_overrides"]) if args.tasks == ["all"] else args.tasks
    deadline = float(os.environ["PIPELINE_DEADLINE"]) if os.environ.get("PIPELINE_DEADLINE") else None

    for tag in args.models:
        path = ensure_converted(str(cfg["models"].get(tag, tag)), resolve_path(cfg["convert_dir"]),
                                resolve_path(cfg["hf_cache"]))
        if args.prepare_only:
            continue
        for task in tasks:
            hp = {**cfg["defaults"], **(cfg.get("task_overrides") or {}).get(task, {})}
            for seed in args.seeds or cfg["seeds"]:
                run_llm_finetune(task_root, task, path, tag, resolve_path(cfg["output_root"]), seed, hp,
                                 deadline=deadline, probe_steps=args.probe_steps)


if __name__ == "__main__":
    main()
