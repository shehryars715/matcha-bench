"""LLM baseline: LoRA fine-tuning of MentalGLM on the MATCHA tasks (one run per task and seed).

    python scripts/run_llm_finetune.py --tasks matcha_disorder_post_masked --models mentalglm-chat
    python scripts/run_llm_finetune.py --tasks all --models mentalglm-chat --seeds 42
    python scripts/run_llm_finetune.py --tasks matcha_user_binary_masked --models mentalglm-chat --probe-steps 6

Finished runs (metrics.json present) are skipped. `--archive-smaller-tests` moves aside the test
results of finished runs that cover fewer test items than the config now asks for (into
test_subset<n>/), so the next call re-tests their saved best adapter without retraining.
`--archive-diverged` moves aside finished runs whose every test answer is unparsed (a NaN
divergence) into diverged/, so the next call trains them again. `--probe-steps N` only times N optimizer steps
and a small generation pass, writing outputs/llm/finetune/_probe/<task>__<model>.json, to plan
GPU time. Same command-line shape as scripts/run_finetune.py, so the pipeline can run either.
"""

import argparse
import gc
import os

import torch

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.llm.convert import ensure_converted
from matcha_bench.llm.train import archive_diverged, archive_smaller_test, planned_test_size, run_llm_finetune


def hparams(cfg: dict, task: str, tag: str) -> dict:
    """defaults < task_overrides[task] < model_overrides[tag] (e.g. a full-data variant of a model)."""
    return {**cfg["defaults"], **(cfg.get("task_overrides") or {}).get(task, {}),
            **(cfg.get("model_overrides") or {}).get(tag, {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--probe-steps", type=int)
    ap.add_argument("--prepare-only", action="store_true", help="download and convert the models, then stop")
    ap.add_argument("--archive-smaller-tests", action="store_true",
                    help="move aside finished runs tested on fewer items than now planned, then stop")
    ap.add_argument("--archive-diverged", action="store_true",
                    help="move aside finished runs with every test answer unparsed (diverged), then stop")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. defaults.epochs=1")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    task_root = resolve_path(cfg["task_root"])
    tasks = sorted(cfg["task_overrides"]) if args.tasks == ["all"] else args.tasks
    deadline = float(os.environ["PIPELINE_DEADLINE"]) if os.environ.get("PIPELINE_DEADLINE") else None

    if args.archive_smaller_tests:  # cheap (no model needed); the runs are then re-tested as usual
        for task in tasks:
            for tag in args.models:
                n = planned_test_size(task_root, task, hparams(cfg, task, tag))
                for seed in args.seeds or cfg["seeds"]:
                    archive_smaller_test(resolve_path(cfg["output_root"]) / task / tag / f"seed{seed}", n)
        return
    if args.archive_diverged:
        for task in tasks:
            for tag in args.models:
                for seed in args.seeds or cfg["seeds"]:
                    archive_diverged(resolve_path(cfg["output_root"]) / task / tag / f"seed{seed}")
        return

    for tag in args.models:
        path = ensure_converted(str(cfg["models"].get(tag, tag)), resolve_path(cfg["convert_dir"]),
                                resolve_path(cfg["hf_cache"]))
        if args.prepare_only:
            continue
        for task in tasks:
            hp = hparams(cfg, task, tag)
            seeds = args.seeds or cfg["seeds"]
            for seed in seeds[:1] if args.probe_steps else seeds:  # timing does not depend on the seed
                recovered = (cfg.get("finished_training") or {}).get(f"{task}/{tag}/seed{seed}")
                run_llm_finetune(task_root, task, path, tag, resolve_path(cfg["output_root"]), seed, hp,
                                 deadline=deadline, probe_steps=args.probe_steps, recovered_history=recovered)
                gc.collect()  # free the previous run's 9B model before loading the next one
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
