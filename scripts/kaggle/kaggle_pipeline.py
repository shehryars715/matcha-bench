"""One-command Kaggle sessions for the full pipeline (pre-training -> fine-tuning -> results).

    python scripts/kaggle/kaggle_pipeline.py start      # start the next session (resumes the last one)
    python scripts/kaggle/kaggle_pipeline.py start --budget-hours 5.5   # shorter session (little GPU quota left)
    python scripts/kaggle/kaggle_pipeline.py start --pipeline configs/pipeline_mentalglm.yaml   # the LLM baseline
    python scripts/kaggle/kaggle_pipeline.py status     # is the current session queued / running / done?
    python scripts/kaggle/kaggle_pipeline.py log        # what each session did (pipeline_log.jsonl)
    python scripts/kaggle/kaggle_pipeline.py results    # download result tables and metrics

    # switch the next sessions to another Kaggle account (FILE holds its access token); it reads
    # the dataset and the earlier sessions from the accounts that own them, which must share both
    python scripts/kaggle/kaggle_pipeline.py account --token-file FILE --data-owner OLD_USERNAME

    # a second, independent chain of sessions (e.g. on another account, with its own share of the
    # runs): every action takes --lane; its sessions are matcha-bench-<lane>-run-001, -002, ...
    python scripts/kaggle/kaggle_pipeline.py start --lane 2 --pipeline configs/kaggle/pipeline_llm_full_2.yaml

Each session (per lane) is a new Kaggle job (matcha-bench-run-001, -002, ...) with a hard time cap
(the pipeline config's budget_hours). It attaches the previous session's output, restores
outputs/ (checkpoints, trained models, finished runs) and continues with scripts/run_pipeline.py,
which stops cleanly before the cap. Session history is kept in outputs/kaggle/pipeline_state.json
(outputs/kaggle/pipeline_state_<lane>.json for a lane).
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
    if args.pipeline is None:
        # Reuse the pipeline of the session being resumed (else the lane's last one).
        by_ref = {ref(r): r for r in state["runs"]}
        prev = by_ref.get(sources[0]) if sources else (state["runs"][-1] if state["runs"] else None)
        args.pipeline = (prev or {}).get("pipeline") or "configs/pipeline_mentalbert.yaml"
    cfg = load_config(resolve_path(args.pipeline))
    n = len(state["runs"]) + 1
    slug = f"{RUN_PREFIX}{n:03d}"
    budget = args.budget_hours or cfg["budget_hours"]
    command = f"python scripts/run_pipeline.py --config {args.pipeline} --budget-hours {budget}"
    if args.margin_minutes is not None:
        command += f" --margin-minutes {args.margin_minutes}"
    url = push_kernel(slug, [command], sources, timeout_hours=budget, data_owner=state.get("data_owner"),
                      extra_datasets=cfg.get("datasets", []))
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
        raise SystemExit("No sessions yet. Run: python scripts/kaggle/kaggle_pipeline.py start")
    return state["runs"][-1]


def status(args) -> None:
    run = current_run()
    print(f"{run['slug']}: {kernel_status(ref(run))}")
    print(run["url"])


def download(run: dict, pattern: str) -> "Path":
    dest = REPO_ROOT / "outputs" / "kaggle" / run["slug"]
    dest.mkdir(parents=True, exist_ok=True)
    # 200 files per page (the most Kaggle allows): a session's output now lists ~500 files, and at the
    # default 20 per page the download is slow enough to time out
    kaggle("kernels", "output", ref(run), "-p", str(dest), "--file-pattern", pattern, "--page-size", "200", "-q", "-o")
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


HF_UPLOAD_SLUG = "matcha-bench-hf-upload"
FINAL_LLM_TAG = "mentalglm-chat-full"
HF_UPLOAD_TEMPLATE = r'''
import glob, json, os
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

REPO_ID = {repo_id!r}
api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
api.create_repo(REPO_ID, private=True, exist_ok=True)
done = 0
for adapter in sorted(glob.glob("/kaggle/input/**/outputs/llm/finetune/*/{tag}/seed*/best_adapter", recursive=True)):
    run_dir = os.path.dirname(adapter)
    if not os.path.exists(os.path.join(run_dir, "metrics.json")):
        print("skip (run not finished):", run_dir)
        continue
    seed, task = os.path.basename(run_dir), os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    path = f"{{task}}/{{seed}}"
    api.upload_folder(repo_id=REPO_ID, folder_path=adapter, path_in_repo=path, commit_message=f"Add {{path}}")
    api.upload_file(repo_id=REPO_ID, path_or_fileobj=os.path.join(run_dir, "metrics.json"),
                    path_in_repo=f"{{path}}/metrics.json", commit_message=f"Add {{path}} metrics")
    print("uploaded", path, flush=True)
    done += 1
print(f"{{done}} adapters uploaded to https://huggingface.co/{{REPO_ID}}")
'''


def hf_upload(args) -> None:
    """Upload the final (mentalglm-chat-full) LoRA adapters of the lane's last session straight from Kaggle
    to HF (CPU job); earlier runs (capped, pilot) and save points are not uploaded."""
    import tempfile

    run = current_run()
    status = kernel_status(ref(run))
    if status != "complete":
        raise SystemExit(f"{ref(run)} is {status}; upload after it completes.")
    user = username()
    meta = {"id": f"{user}/{HF_UPLOAD_SLUG}", "title": HF_UPLOAD_SLUG, "code_file": "upload.py",
            "language": "python", "kernel_type": "script", "is_private": "true", "enable_internet": "true",
            "enable_gpu": "false", "kernel_sources": [ref(run)]}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (tmp / "upload.py").write_text(HF_UPLOAD_TEMPLATE.format(repo_id=args.hf_repo, tag=FINAL_LLM_TAG), encoding="utf-8")
        if kaggle("kernels", "push", "-p", str(tmp)).returncode != 0:
            raise SystemExit("kaggle kernels push failed")
    # Kaggle only hands secrets to interactive runs: this pushed version fails on HF_TOKEN, and the
    # upload happens when the account owner opens the notebook in the editor and clicks Run All
    print(f"Upload notebook ready ({ref(run)} attached). Open it in the editor on {user} and click Run All:")
    print(f"https://www.kaggle.com/code/{user}/{HF_UPLOAD_SLUG}")


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
    ap.add_argument("action", choices=["start", "status", "log", "results", "account", "hf-upload"])
    ap.add_argument("--pipeline", "--config", dest="pipeline", default=None,
                    help="start: pipeline config the session runs (default: the resumed session's, "
                         "else configs/pipeline_mentalbert.yaml)")
    ap.add_argument("--budget-hours", type=float, help="start: session cap instead of the config's budget_hours")
    ap.add_argument("--margin-minutes", type=float,
                    help="start: stop this long before the cap instead of the config's safety_margin_minutes")
    ap.add_argument("--token-file", help="account: file holding that account's Kaggle access token")
    ap.add_argument("--data-owner", help="account: Kaggle user that owns the matcha-bench-data dataset")
    ap.add_argument("--lane", help="an independent chain of sessions with its own state and account")
    ap.add_argument("--hf-repo", default="shehryars715/mentalglm-matcha-lora",
                    help="hf-upload: private Hugging Face repo for the LoRA adapters")
    args = ap.parse_args()
    if args.lane:
        global STATE, RUN_PREFIX
        STATE = STATE.with_name(f"pipeline_state_{args.lane}.json")
        RUN_PREFIX = f"matcha-bench-{args.lane}-run-"
    token_file = load_state().get("token_file")
    if token_file and args.action != "account":
        os.environ["KAGGLE_API_TOKEN"] = token_file  # the kaggle CLI subprocesses inherit it
    {"start": start, "status": status, "log": log, "results": results, "account": account,
     "hf-upload": hf_upload}[args.action](args)


if __name__ == "__main__":
    main()
