"""One-command Kaggle sessions for the full pipeline (pre-training -> fine-tuning -> results).

    python scripts/kaggle_pipeline.py start      # start the next session (resumes the last one)
    python scripts/kaggle_pipeline.py status     # is the current session queued / running / done?
    python scripts/kaggle_pipeline.py log        # what each session did (pipeline_log.jsonl)
    python scripts/kaggle_pipeline.py results    # download result tables and metrics

Each session is a new Kaggle job (matcha-bench-run-001, -002, ...) with a hard time cap
(configs/pipeline.yaml: budget_hours). It attaches the previous session's output, restores
outputs/ (checkpoints, trained models, finished runs) and continues with scripts/run_pipeline.py,
which stops cleanly before the cap. Session history is kept in outputs/kaggle/pipeline_state.json.
"""

import argparse
import json
import subprocess
import sys
import time

import _bootstrap  # noqa: F401
from kaggle_job import kaggle, push_kernel, username
from matcha_bench.config import REPO_ROOT, load_config, resolve_path

STATE = REPO_ROOT / "outputs" / "kaggle" / "pipeline_state.json"
RUN_PREFIX = "matcha-bench-run-"


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"runs": []}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def kernel_status(slug: str) -> str:
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", f"{username()}/{slug}"],
                         capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    for word in ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCEL"):
        if word in text.upper():
            return word.lower()
    return text or "unknown"


def start(args) -> None:
    cfg = load_config(resolve_path(args.config))
    state = load_state()
    sources = []
    if state["runs"]:
        last = state["runs"][-1]["slug"]
        status = kernel_status(last)
        if status in ("queued", "running"):
            raise SystemExit(f"{last} is still {status}; wait for it to finish first.")
        if status != "complete":
            print(f"warning: last session {last} ended with status '{status}'.")
            good = [r["slug"] for r in state["runs"] if kernel_status(r["slug"]) == "complete"]
            if not good:
                print("No completed session to resume from; starting from scratch.")
            last = good[-1] if good else None
        if last:
            sources = [last]
    n = len(state["runs"]) + 1
    slug = f"{RUN_PREFIX}{n:03d}"
    url = push_kernel(slug, ["python scripts/run_pipeline.py"], sources, timeout_hours=cfg["budget_hours"])
    state["runs"].append({"slug": slug, "resumes": sources[0] if sources else None,
                          "pushed": time.strftime("%Y-%m-%d %H:%M:%S"), "url": url})
    save_state(state)
    print(f"Started session {n} ({slug}), hard cap {cfg['budget_hours']} h"
          + (f", resuming from {sources[0]}" if sources else ", fresh start"))
    print(url)


def current_slug() -> str:
    state = load_state()
    if not state["runs"]:
        raise SystemExit("No sessions yet. Run: python scripts/kaggle_pipeline.py start")
    return state["runs"][-1]["slug"]


def status(args) -> None:
    slug = current_slug()
    print(f"{slug}: {kernel_status(slug)}")
    print(load_state()["runs"][-1]["url"])


def download(slug: str, pattern: str) -> "Path":
    dest = REPO_ROOT / "outputs" / "kaggle" / slug
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{username()}/{slug}", "-p", str(dest), "--file-pattern", pattern, "-q", "-o")
    return dest


def log(args) -> None:
    slug = current_slug()
    dest = download(slug, r"pipeline_log\.jsonl$")
    files = list(dest.rglob("pipeline_log.jsonl"))
    if not files:
        print("No pipeline log yet (session not finished, or it failed before starting).")
        return
    for line in files[0].read_text(encoding="utf-8").splitlines():
        print(line)


def results(args) -> None:
    slug = current_slug()
    dest = download(slug, r"(outputs/results/|metrics\.json$|run_info\.json$|pipeline_log\.jsonl$)")
    print(f"Downloaded result files to {dest}")
    summary = list(dest.rglob("summary_f1.csv"))
    if summary:
        print(summary[0].read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["start", "status", "log", "results"])
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()
    {"start": start, "status": status, "log": log, "results": results}[args.action](args)


if __name__ == "__main__":
    main()
