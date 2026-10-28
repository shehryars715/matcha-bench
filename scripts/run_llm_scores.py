"""Label probabilities from the fine-tuned MentalGLM runs (no generation; see matcha_bench.llm.score).

    python scripts/run_llm_scores.py --tasks matcha_disorder_post --models mentalglm-chat-full --seeds 42
    python scripts/run_llm_scores.py --tasks matcha_user_binary_masked --models mentalglm-chat --set defaults.limit=200

Each run loads the finished fine-tuning run's best adapter (adapter_root/<task>/<tag>/seed<k>) and
scores test, dev and train in that order; the test arg-max is checked against the run's generated
answers first. Results: outputs/llm/scores/<task>/<tag>/seed<k>/{test,dev,train}_scores.npz and
metrics.json. Same command-line shape as scripts/run_llm_finetune.py, so the pipeline's finetune
stage schedules it one run per GPU; an unfinished split resumes from its last saved chunk.
"""

import argparse
import gc
import os

import torch

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.llm.convert import ensure_converted
from matcha_bench.llm.score import run_llm_scores
from run_llm_finetune import hparams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llm_scores.yaml")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--prepare-only", action="store_true", help="download and convert the models, then stop")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. defaults.limit=200")
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
            for seed in args.seeds or cfg["seeds"]:
                run_llm_scores(task_root, task, path, tag, resolve_path(cfg["adapter_root"]),
                               resolve_path(cfg["output_root"]), seed, hparams(cfg, task, tag),
                               splits=cfg["splits"], deadline=deadline)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
