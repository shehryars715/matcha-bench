"""Run pipeline steps on Kaggle GPUs from the command line (uses the `kaggle` CLI and ~/.kaggle/kaggle.json).

    # once: upload the data zip as a private Kaggle dataset
    python scripts/kaggle/kaggle_job.py upload-data --zip ../matcha-bench-data.zip

    # push a GPU job (runs headless, up to 12 h); each --cmd runs from the repo root
    python scripts/kaggle/kaggle_job.py push --name smoke --cmd "python scripts/run_dapt.py --config configs/dapt/smoke.yaml"
    python scripts/kaggle/kaggle_job.py push --name dapt-random --cmd "python scripts/run_dapt.py --config configs/dapt/random.yaml"

    # fine-tune using the models produced by earlier jobs (their outputs are attached and merged)
    python scripts/kaggle/kaggle_job.py push --name ft-mentalbert --inputs-from dapt-random dapt-depression \\
        --cmd "python scripts/run_finetune.py --tasks suicide cognitive --models all"

    python scripts/kaggle/kaggle_job.py status --name smoke
    python scripts/kaggle/kaggle_job.py output --name smoke          # -> outputs/kaggle/smoke/

Each job clones the public GitHub repo, links the dataset's data/ and lexicons/, copies outputs/
from any --inputs-from jobs (so DAPT resumes and fine-tuning finds the models), runs the
commands, and leaves everything under /kaggle/working/matcha-bench/outputs as the job output.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from matcha_bench.config import REPO_ROOT

GIT_URL = "https://github.com/shehryars715/matcha-bench.git"
DATASET_SLUG = "matcha-bench-data"
PREFIX = "matcha-bench-"

JOB_TEMPLATE = r'''
import glob, os, shutil, subprocess, sys, time, zipfile

os.environ["JOB_START"] = str(time.time())  # time budgets count from here, including setup
CONFIG = {config}
REPO = "/kaggle/working/matcha-bench"


def sh(cmd, cwd=None):
    print(f"\n$ {{cmd}}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
sh(f"git clone --depth 1 --quiet {{CONFIG['git_url']}} {{REPO}}")
sh("git log -1 --format='code version: %h %s'", cwd=REPO)
sh("pip install -q jieba")

# Locate the data: an extracted data/dapt folder, or the uploaded zip.
data_root = next((os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/data/dapt", recursive=True)), None)
if data_root is None:
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if not zips:
        sys.exit("No data found under /kaggle/input (attach the matcha-bench-data dataset)")
    data_root = "/kaggle/working/input_data"
    zipfile.ZipFile(zips[0]).extractall(data_root)
print("data root:", data_root, flush=True)
for name in ("data", "lexicons"):
    target = os.path.join(REPO, name)
    shutil.rmtree(target, ignore_errors=True)
    os.symlink(os.path.join(data_root, name), target)

# Merge outputs/ from earlier jobs (resume DAPT, find trained models, skip finished runs).
for prev in glob.glob("/kaggle/input/**/matcha-bench/outputs", recursive=True):
    print("restoring outputs from", prev, flush=True)
    shutil.copytree(prev, os.path.join(REPO, "outputs"), dirs_exist_ok=True)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for cmd in CONFIG["commands"]:
    sh(cmd, cwd=REPO)

# Keep the output small: drop DAPT checkpoints once a final model exists.
for ckpt in glob.glob(os.path.join(REPO, "outputs/dapt/*/checkpoint-*")):
    if os.path.exists(os.path.join(os.path.dirname(ckpt), "final")):
        shutil.rmtree(ckpt)
shutil.rmtree("/kaggle/working/input_data", ignore_errors=True)
sh("du -sh outputs/* || true", cwd=REPO)
'''


def kaggle(*args: str) -> subprocess.CompletedProcess:
    # UTF-8 mode: the CLI writes kernel logs with the default encoding, which fails on Windows (cp1252)
    return subprocess.run([sys.executable, "-m", "kaggle", *args], text=True,
                          env={**os.environ, "PYTHONUTF8": "1"})


def username() -> str:
    if os.environ.get("KAGGLE_API_TOKEN"):  # an access token (or a file holding one) picks another account
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api.config_values["username"]
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists():
        raise SystemExit(f"Missing {cfg} (Kaggle → Settings → API → Create New Token)")
    return json.loads(cfg.read_text(encoding="utf-8"))["username"]


def upload_data(args) -> None:
    zip_path = Path(args.zip).resolve()
    if not zip_path.exists():
        raise SystemExit(f"{zip_path} not found; run scripts/kaggle/pack_kaggle_data.py first")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta = {"title": DATASET_SLUG, "id": f"{username()}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        shutil.copy2(zip_path, tmp / zip_path.name)
        if args.new_version:
            kaggle("datasets", "version", "-p", str(tmp), "-m", args.message)
        else:
            kaggle("datasets", "create", "-p", str(tmp))  # private by default


def push_kernel(slug: str, commands: list[str], kernel_sources: list[str], gpu: str = "NvidiaTeslaT4",
                timeout_hours: float | None = None, data_owner: str | None = None) -> str:
    """Push and start a headless Kaggle job; returns its URL.

    kernel_sources may be bare slugs (this account) or "owner/slug" (shared from another account);
    data_owner is the account that holds the dataset (default: this one).
    """
    user = username()
    config = {"git_url": GIT_URL, "commands": commands}
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_internet": "true",
        "dataset_sources": [f"{data_owner or user}/{DATASET_SLUG}"],
        "kernel_sources": [s if "/" in s else f"{user}/{s}" for s in kernel_sources],
    }
    meta["enable_gpu"] = "true" if gpu else "false"
    if gpu:
        meta["machine_shape"] = gpu
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (tmp / "job.py").write_text(JOB_TEMPLATE.format(config=json.dumps(config)), encoding="utf-8")
        extra = ["-t", str(int(timeout_hours * 3600))] if timeout_hours else []
        result = kaggle("kernels", "push", "-p", str(tmp), *extra)
        if result.returncode != 0:
            raise SystemExit("kaggle kernels push failed")
    return f"https://www.kaggle.com/code/{user}/{slug}"


def push(args) -> None:
    url = push_kernel(PREFIX + args.name, args.cmd, [PREFIX + n for n in args.inputs_from], args.gpu, args.timeout_hours)
    print(url)


def status(args) -> None:
    kaggle("kernels", "status", f"{username()}/{PREFIX}{args.name}")


def output(args) -> None:
    dest = REPO_ROOT / "outputs" / "kaggle" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{username()}/{PREFIX}{args.name}", "-p", str(dest))
    print(f"downloaded to {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)

    up = sub.add_parser("upload-data")
    up.add_argument("--zip", default=str(REPO_ROOT.parent / "matcha-bench-data.zip"))
    up.add_argument("--new-version", action="store_true")
    up.add_argument("--message", default="update data")

    p = sub.add_parser("push")
    p.add_argument("--name", required=True)
    p.add_argument("--cmd", action="append", required=True)
    p.add_argument("--inputs-from", nargs="*", default=[])
    p.add_argument("--gpu", default="NvidiaTeslaT4", help="Kaggle machine_shape; '' for CPU")
    p.add_argument("--timeout-hours", type=float, help="hard cap on the job's run time")

    for action in ("status", "output"):
        s = sub.add_parser(action)
        s.add_argument("--name", required=True)

    args = ap.parse_args()
    {"upload-data": upload_data, "push": push, "status": status, "output": output}[args.action](args)


if __name__ == "__main__":
    main()
