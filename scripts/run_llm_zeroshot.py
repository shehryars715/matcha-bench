"""Zero-shot LLM scores on the MATCHA tasks: the released model, no MATCHA training.

    python scripts/run_llm_zeroshot.py --tasks matcha_user_binary_masked --models mentalglm-chat
    python scripts/run_llm_zeroshot.py --tasks all --models mentalglm-chat glm-4-9b-chat

Prompts, inputs (characters kept per task), 4-bit weights, greedy decoding and answer parsing are
those of the fine-tuned runs (configs/llm.yaml), except a longer answer cap (configs/llm_zeroshot.yaml);
unparsed answers count as wrong. Decoding is
deterministic, so there is one run per model (seed 0 in the folder name). Results:
outputs/llm/zeroshot/<task>/<model>/seed0/metrics.json. Same command-line shape as
scripts/run_llm_finetune.py, so the pipeline's finetune stage schedules it one run per GPU.
"""

import argparse
import gc
import json
import os
import shutil

import torch

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.llm.convert import ensure_converted
from matcha_bench.llm.train import run_llm_zeroshot
from run_llm_finetune import hparams


def archive_other_cap(out_dir, max_new_tokens: int) -> bool:
    """Move a finished run's results into cap<n>/ when they were generated with another answer
    length cap, so the run counts as unfinished. Returns True if it archived anything."""
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    old = json.loads(metrics_path.read_text(encoding="utf-8"))["hyperparameters"]["max_new_tokens"]
    if old == max_new_tokens:
        return False
    archive = out_dir / f"cap{old}"
    archive.mkdir(exist_ok=True)
    for name in ("test_predictions.npz", "test_answers.jsonl", "metrics.json"):  # metrics.json last
        if (out_dir / name).exists():
            shutil.move(str(out_dir / name), str(archive / name))
    print(f"{out_dir}: generated with max_new_tokens={old}, archived to {archive.name}; rerun with {max_new_tokens}",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llm_zeroshot.yaml")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--prepare-only", action="store_true", help="download and convert the models, then stop")
    ap.add_argument("--archive-other-cap", action="store_true",
                    help="move aside finished runs generated with a different max_new_tokens (into "
                         "cap<n>/), so the next call redoes them; then stop")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. defaults.eval_batch_size=8")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    task_root = resolve_path(cfg["task_root"])
    tasks = sorted(cfg["task_overrides"]) if args.tasks == ["all"] else args.tasks
    deadline = float(os.environ["PIPELINE_DEADLINE"]) if os.environ.get("PIPELINE_DEADLINE") else None

    if args.archive_other_cap:
        for task in tasks:
            for tag in args.models:
                for seed in args.seeds or cfg["seeds"]:
                    out_dir = resolve_path(cfg["output_root"]) / task / tag / f"seed{seed}"
                    archive_other_cap(out_dir, hparams(cfg, task, tag)["max_new_tokens"])
        return

    for tag in args.models:
        path = ensure_converted(str(cfg["models"].get(tag, tag)), resolve_path(cfg["convert_dir"]),
                                resolve_path(cfg["hf_cache"]))
        if args.prepare_only:
            continue
        for task in tasks:
            for seed in args.seeds or cfg["seeds"]:
                run_llm_zeroshot(task_root, task, path, tag, resolve_path(cfg["output_root"]), seed,
                                 hparams(cfg, task, tag), deadline=deadline)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
