"""LLM sanity check: label MentalGLM's own test sets with a released MentalGLM model and compare
with the paper (MentalGLM-chat: SOS-HL-1K F1 85.12, SocialCD-3K micro-F1 71.04).

    python scripts/run_llm_eval.py                       # every run listed under sanity.runs
    python scripts/run_llm_eval.py --model mentalglm-chat --quant 4bit --tasks suicide
    python scripts/run_llm_eval.py --prepare-only        # download and convert the models only

Results: outputs/llm/sanity/<model>__<quant>[__bs<n>]/<task>/{metrics.json, answers.jsonl}; a run
with its own batch_size gets the __bs suffix. A finished (run, task) is skipped.
"""

import argparse
import json

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.llm.convert import ensure_converted
from matcha_bench.llm.data import load_mentalglm_test
from matcha_bench.llm.infer import evaluate_texts, load_model
from matcha_bench.llm.prompts import get_prompt
from matcha_bench.utils import get_logger, save_json

log = get_logger()
TASK_META = {  # label names and averaging as in the paper's Table 3
    "suicide": (["low_risk", "high_risk"], "binary"),
    "cognitive": (["all_or_nothing", "overgeneralization", "mental_filter", "disqualifying_positive",
                   "mind_reading", "fortune_telling", "magnification", "emotional_reasoning",
                   "should_statements", "labeling", "blaming_oneself", "blaming_others"], "micro"),
}
def sanity_run_name(run: dict) -> str:
    """Folder name of a sanity run; shared with the pipeline's done-check."""
    return f"{run['model']}__{run['quant']}" + (f"__bs{run['batch_size']}" if run.get("batch_size") else "")


PAPER = {"mentalglm-chat": {"suicide": 85.12, "cognitive": 71.04}, "mentalglm": {"suicide": 79.20, "cognitive": 73.06}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/llm.yaml")
    ap.add_argument("--model", help="model tag (default: every run under sanity.runs)")
    ap.add_argument("--quant", choices=["fp16", "4bit", "fp32"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--limit", type=int, help="label only the first N test items (smoke test)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    sc = cfg["sanity"]
    runs = [{"model": args.model, "quant": args.quant or "fp16"}] if args.model else sc["runs"]
    tasks = args.tasks or sc["tasks"]
    out_root = resolve_path(sc["output_root"])

    for run in runs:
        tag, quant = run["model"], run["quant"]
        name = sanity_run_name(run) + (f"__limit{args.limit}" if args.limit else "")
        batch_size = args.batch_size or run.get("batch_size") or sc["batch_size"]
        todo = [t for t in tasks if not (out_root / name / t / "metrics.json").exists()]
        if not todo and not args.prepare_only:
            log.info("Skipping %s (done)", name)
            continue
        path = ensure_converted(str(cfg["models"].get(tag, tag)), resolve_path(cfg["convert_dir"]),
                                resolve_path(cfg["hf_cache"]))
        if args.prepare_only:
            continue
        model, tokenizer = load_model(path, quant)
        for task in todo:
            df = load_mentalglm_test(task, resolve_path(sc["data_cache"]))
            if args.limit:
                df = df.head(args.limit)
            label_names, average = TASK_META[task]
            res = evaluate_texts(model, tokenizer, get_prompt(task), df["text"].tolist(), df["label"].tolist(),
                                 label_names, average, batch_size=batch_size,
                                 max_new_tokens=sc["max_new_tokens"])
            m = res["metrics"]
            m.update({"model_tag": tag, "model_path": path, "quant": quant, "task": task, "batch_size": batch_size,
                      "paper_f1": PAPER.get(tag, {}).get(task)})
            out = out_root / name / task
            save_json(m, out / "metrics.json")
            with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
                for a, p in zip(res["answers"], res["preds"].tolist()):
                    f.write(json.dumps({"answer": a, "pred": p}, ensure_ascii=False) + "\n")
            log.info("%s | %s | F1 %.2f (paper %s) P %.2f R %.2f | unparsed %d | %.2f items/s", name, task,
                     100 * m["f1"], m["paper_f1"], 100 * m["precision"], 100 * m["recall"], m["unparsed"],
                     m["items_per_second"])
        del model
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
