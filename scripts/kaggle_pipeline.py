"""One-command Kaggle sessions for the full pipeline (pre-training -> fine-tuning -> results).

    python scripts/kaggle_pipeline.py start      # start the next session (resumes the last one)
    python scripts/kaggle_pipeline.py start --budget-hours 5.5   # shorter session (little GPU quota left)
    python scripts/kaggle_pipeline.py start --pipeline configs/pipeline_llm.yaml   # the LLM baseline
    python scripts/kaggle_pipeline.py status     # is the current session queued / running / done?
    python scripts/kaggle_pipeline.py log        # what each session did (pipeline_log.jsonl)
    python scripts/kaggle_pipeline.py results    # download result tables and metrics

    # switch the next sessions to another Kaggle account (FILE holds its access token); it reads
    # the dataset and the earlier sessions from the accounts that own them, which must share both
    python scripts/kaggle_pipeline.py account --token-file FILE --data-owner OLD_USERNAME

Each session is a new Kaggle job (matcha-bench-run-001, -002, ...) with a hard time cap
(configs/pipeline.yaml: budget_hours). It attaches the previous session's output, restores
outputs/ (checkpoints, trained models, finished runs) and continues with scripts/run_pipeline.py,
which stops cleanly before the cap. Session history is kept in outputs/kaggle/pipeline_state.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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


def ref(run: dict) -> str:
    """owner/slug of a session (sessions from before the account switch have no owner: this account)."""
    return f"{run.get('owner') or username()}/{run['slug']}"


def kernel_status(kernel: str) -> str:
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", kernel],
                         capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    for word in ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCEL"):
        if word in text.upper():
            return word.lower()
    return text or "unknown"


def start(args) -> None:
    cfg = load_config(resolve_path(args.pipeline))
    state = load_state()
    sources = []
    if state["runs"]:
        last = ref(state["runs"][-1])
        status = kernel_status(last)
        if status in ("queued", "running"):
            raise SystemExit(f"{last} is still {status}; wait for it to finish first.")
        if status != "complete":
            print(f"warning: last session {last} ended with status '{status}'.")
            good = [ref(r) for r in state["runs"] if kernel_status(ref(r)) == "complete"]
            if not good:
                print("No completed session to resume from; starting from scratch.")
            last = good[-1] if good else None
        if last:
            sources = [last]
    n = len(state["runs"]) + 1
    slug = f"{RUN_PREFIX}{n:03d}"
    budget = args.budget_hours or cfg["budget_hours"]
    command = f"python scripts/run_pipeline.py --config {args.pipeline} --budget-hours {budget}"
    url = push_kernel(slug, [command], sources, timeout_hours=budget, data_owner=state.get("data_owner"))
    state["runs"].append({"slug": slug, "owner": username(), "pipeline": args.pipeline,
                          "resumes": sources[0] if sources else None,
                          "pushed": time.strftime("%Y-%m-%d %H:%M:%S"), "url": url})
    save_state(state)
    print(f"Started session {n} ({slug}, {args.pipeline}), hard cap {budget} h"
          + (f", resuming from {sources[0]}" if sources else ", fresh start"))
    print(url)


def current_run() -> dict:
    state = load_state()
    if not state["runs"]:
        raise SystemExit("No sessions yet. Run: python scripts/kaggle_pipeline.py start")
    return state["runs"][-1]


def status(args) -> None:
    run = current_run()
    print(f"{run['slug']}: {kernel_status(ref(run))}")
    print(run["url"])


def download(run: dict, pattern: str) -> "Path":
    dest = REPO_ROOT / "outputs" / "kaggle" / run["slug"]
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref(run), "-p", str(dest), "--file-pattern", pattern, "-q", "-o")
    return dest


def log(args) -> None:
    dest = download(current_run(), r"pipeline_log\.jsonl$")
    files = list(dest.rglob("pipeline_log.jsonl"))
    if not files:
        print("No pipeline log yet (session not finished, or it failed before starting).")
        return
    for line in files[0].read_text(encoding="utf-8").splitlines():
        print(line)


def results(args) -> None:
    dest = download(current_run(), r"(outputs/results/|outputs/llm/.*\.jsonl?$|metrics\.json$|run_info\.json$|pipeline_log\.jsonl$)")
    print(f"Downloaded result files to {dest}")
    summary = list(dest.rglob("summary_f1.csv"))
    if summary:
        print(summary[0].read_text(encoding="utf-8"))


def account(args) -> None:
    state = load_state()
    if state.get("token_file"):
        os.environ["KAGGLE_API_TOKEN"] = state["token_file"]
    for run in state["runs"]:  # pin the owner of earlier sessions before the account changes
        run.setdefault("owner", username())
    if args.token_file:
        state["token_file"] = str(Path(args.token_file).resolve())
        os.environ["KAGGLE_API_TOKEN"] = state["token_file"]
    if args.data_owner:
        state["data_owner"] = args.data_owner
    save_state(state)
    print(f"account: {username()}  data: {state.get('data_owner') or username()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["start", "status", "log", "results", "account"])
    ap.add_argument("--pipeline", "--config", dest="pipeline", default="configs/pipeline.yaml",
                    help="start: pipeline config the session runs (configs/pipeline_llm.yaml for the LLM baseline)")
    ap.add_argument("--budget-hours", type=float, help="start: session cap instead of the config's budget_hours")
    ap.add_argument("--token-file", help="account: file holding that account's Kaggle access token")
    ap.add_argument("--data-owner", help="account: Kaggle user that owns the matcha-bench-data dataset")
    args = ap.parse_args()
    token_file = load_state().get("token_file")
    if token_file and args.action != "account":
        os.environ["KAGGLE_API_TOKEN"] = token_file  # the kaggle CLI subprocesses inherit it
    {"start": start, "status": status, "log": log, "results": results, "account": account}[args.action](args)


if __name__ == "__main__":
    main()
