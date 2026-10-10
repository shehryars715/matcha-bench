"""Run the whole GPU pipeline inside one time-limited session, resuming where the last one stopped.

    python scripts/run_pipeline.py --deadline <unix-seconds>      # used by Kaggle jobs
    python scripts/run_pipeline.py --budget-hours 9                # deadline = now + budget - margin

Stages (configs/pipeline.yaml) run in order:
* dapt: skipped if outputs/dapt/<variant>/final exists; otherwise trains (resuming from the last
  checkpoint) until done or the deadline, then the pipeline stops for this session;
* finetune: one subprocess per (task, model, seed); finished runs are skipped, and a run is only
  started if its reserved minutes fit before the deadline;
* tfidf, summary: quick CPU steps once everything else is done.
The session log of what happened is appended to outputs/pipeline_log.jsonl.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from matcha_bench.config import REPO_ROOT, load_config, resolve_path

PY = sys.executable


def log_event(**event) -> None:
    event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[pipeline] {event}", flush=True)
    path = REPO_ROOT / "outputs" / "pipeline_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def minutes_left(deadline: float) -> float:
    return (deadline - time.time()) / 60


def stage_preflight(stage, deadline) -> bool:
    marker = REPO_ROOT / "outputs" / ".preflight_ok"
    if marker.exists():
        return True
    code = run([PY, "scripts/run_dapt.py", "--config", "configs/dapt/smoke.yaml",
                "--set", "output_dir=outputs/dapt/_preflight"])
    if code != 0:
        log_event(stage="preflight", status="failed", returncode=code)
        raise SystemExit("preflight failed; see the log above")
    marker.write_text("ok", encoding="utf-8")
    return True


def stage_dapt(stage, deadline) -> bool:
    cfg = load_config(resolve_path(stage["config"]))
    out = resolve_path(cfg["output_dir"])
    if (out / "final").exists():
        log_event(stage=stage["name"], status="already done")
        return True
    if minutes_left(deadline) < 15:
        log_event(stage=stage["name"], status="not started (time)")
        return False
    code = run([PY, "scripts/run_dapt.py", "--config", stage["config"], "--set", f"deadline={deadline}"])
    done = (out / "final").exists()
    progress = out / "progress.json"
    step = json.loads(progress.read_text(encoding="utf-8")).get("global_step") if progress.exists() else None
    log_event(stage=stage["name"], status="done" if done else "paused", returncode=code, step=step)
    if code != 0 and not done:
        raise SystemExit(f"{stage['name']} failed with code {code}")
    return done


def stage_finetune(stage, deadline) -> bool:
    ft = load_config(resolve_path(stage["config"]))
    out_root = resolve_path(ft["output_root"])
    reserve = stage.get("minutes_per_run", {})
    completed_all = True
    for task in stage["tasks"]:
        if not (resolve_path(ft["task_root"]) / task / "meta.json").exists():
            continue
        for model in stage["models"]:
            source = str(ft["models"].get(model, model))
            if source.startswith("outputs/") and not resolve_path(source).exists():
                completed_all = False  # its DAPT model is not trained yet
                continue
            for seed in ft["seeds"]:
                if (out_root / task / model / f"seed{seed}" / "metrics.json").exists():
                    continue
                need = reserve.get(task, reserve.get("default", 20))
                if minutes_left(deadline) < need:
                    log_event(stage="finetune", status="paused (time)", next=f"{task}/{model}/seed{seed}")
                    return False
                code = run([PY, "scripts/run_finetune.py", "--tasks", task, "--models", model, "--seeds", str(seed)])
                log_event(stage="finetune", run=f"{task}/{model}/seed{seed}", returncode=code)
                if code != 0:
                    completed_all = False
    log_event(stage="finetune", status="done" if completed_all else "done (some runs failed or waiting)")
    return completed_all


def stage_tfidf(stage, deadline) -> bool:
    return run([PY, "scripts/run_tfidf.py"]) == 0


def stage_summary(stage, deadline) -> bool:
    run([PY, "scripts/summarize_results.py"])
    run([PY, "scripts/summarize_results.py", "--compare", "matcha-multidisorder", "matcha-random"])
    run([PY, "scripts/summarize_results.py", "--compare", "matcha-depression", "matcha-random"])
    return True


STAGES = {
    "preflight": stage_preflight,
    "dapt": stage_dapt,
    "finetune": stage_finetune,
    "tfidf": stage_tfidf,
    "summary": stage_summary,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--deadline", type=float, help="unix time by which to stop (already includes the margin)")
    ap.add_argument("--budget-hours", type=float, help="override the config budget (from now)")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))

    if args.deadline:
        deadline = args.deadline
    else:
        budget = args.budget_hours or cfg["budget_hours"]
        start = float(os.environ.get("JOB_START", time.time()))  # set by Kaggle jobs at session start
        deadline = start + budget * 3600 - cfg["safety_margin_minutes"] * 60
    log_event(stage="session", status="start", minutes_available=round(minutes_left(deadline)))

    for stage in cfg["stages"]:
        if not STAGES[stage["type"]](stage, deadline):
            log_event(stage="session", status="paused", at=stage["name"], minutes_left=round(minutes_left(deadline)))
            print("\nSession ended before the pipeline finished. Start the next session to continue.")
            return
    (REPO_ROOT / "outputs" / "PIPELINE_COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log_event(stage="session", status="pipeline complete")


if __name__ == "__main__":
    main()
