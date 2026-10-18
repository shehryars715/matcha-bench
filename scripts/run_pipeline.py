"""Run the whole GPU pipeline inside one time-limited session, resuming where the last one stopped.

    python scripts/run_pipeline.py --deadline <unix-seconds>      # used by Kaggle jobs
    python scripts/run_pipeline.py --budget-hours 9                # deadline = now + budget - margin
    python scripts/run_pipeline.py --no-limit                       # local GPU: run until done
    python scripts/run_pipeline.py --no-limit --dapt-set training.per_device_train_batch_size=16         --finetune-set defaults.batch_size=8 defaults.grad_accum=2   # smaller GPUs (same effective batch)

Stages (configs/pipeline.yaml) run in order:
* dapt: skipped if outputs/dapt/<variant>/final exists; otherwise trains (resuming from the last
  checkpoint) until done or the deadline, then the pipeline stops for this session;
* finetune (one stage per task group, each with its own tasks and models): one subprocess per
  (task, model, seed); finished runs are skipped, and a run is only started if its reserved
  minutes fit before the deadline. With `finetune_one_run_per_gpu: true` and several GPUs, each
  GPU runs its own (task, model, seed) at the same time (output in outputs/logs/finetune/);
  otherwise runs go one at a time across all GPUs;
* tfidf, summary: quick CPU steps once everything else is done. The summary writes the results
  tables and one paired-bootstrap comparison per model pair listed under `compare`.
LLM baseline (configs/pipeline_llm.yaml):
* pip: installs extra packages (peft, bitsandbytes) into the session;
* llm_sanity: scripts/run_llm_eval.py, the released MentalGLM on its own test sets;
* llm_probe: times a few LoRA training steps per task (scripts/run_llm_finetune.py --probe-steps);
* finetune with `script: scripts/run_llm_finetune.py`: the LoRA runs, scheduled like the BERT runs.
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
EXTRA = {"dapt": [], "finetune": []}  # extra --set overrides passed to the step scripts
ONE_RUN_PER_GPU = False               # set from configs/pipeline.yaml in main()
POLL_SECONDS = 10                     # how often parallel fine-tuning runs are checked


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
    code = run([PY, "scripts/run_dapt.py", "--config", stage["config"], "--set", f"deadline={deadline}", *EXTRA["dapt"]])
    done = (out / "final").exists()
    progress = out / "progress.json"
    step = json.loads(progress.read_text(encoding="utf-8")).get("global_step") if progress.exists() else None
    log_event(stage=stage["name"], status="done" if done else "paused", returncode=code, step=step)
    if code != 0 and not done:
        raise SystemExit(f"{stage['name']} failed with code {code}")
    return done


def gpu_slots() -> list[str | None]:
    """CUDA_VISIBLE_DEVICES value for each concurrent fine-tuning run.

    With `finetune_one_run_per_gpu` and more than one GPU, every GPU runs its own (task, model,
    seed) with a true per-GPU batch, instead of one run split across GPUs by DataParallel (which
    leaves small batches waiting on GPU synchronisation). Otherwise there is one slot and the run
    sees every GPU, as before.
    """
    if not ONE_RUN_PER_GPU:
        return [None]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] if n > 1 else [None]


def start_finetune(task: str, model: str, seed: int, device: str | None, script: str = "scripts/run_finetune.py",
                   config: str | None = None, deadline: float | None = None):
    """Launch one fine-tuning run. In parallel mode its output goes to outputs/logs/finetune/."""
    cmd = [PY, script, "--tasks", task, "--models", model, "--seeds", str(seed)]
    cmd += ["--config", config] if config else []
    cmd += ["--set", *EXTRA["finetune"]] if EXTRA["finetune"] else []
    env, log_file = dict(os.environ), None
    if deadline is not None:  # long runs (LLM) stop themselves instead of being killed with the session
        env["PIPELINE_DEADLINE"] = str(deadline)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
        log_dir = REPO_ROOT / "outputs" / "logs" / "finetune"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_").replace("\\", "_")  # tags may also be local paths
        log_file = open(log_dir / f"{task}__{safe_model}__seed{seed}.log", "w", encoding="utf-8")
    print("\n$ " + " ".join(cmd) + (f"   [GPU {device}]" if device is not None else ""), flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file,
                            stderr=subprocess.STDOUT if log_file else None)
    return proc, log_file


def stage_finetune(stage, deadline) -> bool:
    ft = load_config(resolve_path(stage["config"]))
    out_root = resolve_path(ft["output_root"])
    script = stage.get("script", "scripts/run_finetune.py")
    extra = {} if script == "scripts/run_finetune.py" else {"script": script, "config": stage["config"],
                                                             "deadline": deadline}
    reserve = stage.get("minutes_per_run", {})
    if stage.get("before") and run([PY, *stage["before"]]) != 0:  # e.g. mark runs to re-test
        log_event(stage=stage["name"], status="before failed")
        return False
    completed_all = True
    queue = []
    for task in stage["tasks"]:
        if not (resolve_path(ft["task_root"]) / task / "meta.json").exists():
            continue
        for model in stage["models"]:
            source = str(ft["models"].get(model, model))
            if source.startswith("outputs/") and not resolve_path(source).exists():
                completed_all = False  # its DAPT model is not trained yet
                continue
            for seed in ft["seeds"]:
                if not (out_root / task / model / f"seed{seed}" / "metrics.json").exists():
                    queue.append((task, model, seed))

    slots = gpu_slots()
    running: dict[int, tuple] = {}  # slot index -> (job, process, log file)
    paused, prepared = False, False
    while running or (queue and not paused):
        for i, device in enumerate(slots):
            if i in running or not queue or paused:
                continue
            # the first queued run that still fits; a long run waiting for the next session does not
            # block shorter ones behind it
            fits = [job for job in queue if minutes_left(deadline) >= reserve.get(job[0], reserve.get("default", 20))]
            if not fits:
                task, model, seed = queue[0]
                log_event(stage=stage["name"], status="paused (time)", next=f"{task}/{model}/seed{seed}")
                paused = True
                break
            task, model, seed = job = fits[0]
            if extra and stage.get("prepare") and not prepared:  # e.g. download/convert the LLM once
                if run([PY, *stage["prepare"]]) != 0:
                    log_event(stage=stage["name"], status="prepare failed")
                    return False
                prepared = True
            queue.remove(job)
            running[i] = (job, *start_finetune(task, model, seed, device, **extra))
        time.sleep(POLL_SECONDS if running else 0)
        for i in [i for i, (_, proc, _) in running.items() if proc.poll() is not None]:
            (task, model, seed), proc, log_file = running.pop(i)
            if log_file:
                log_file.close()
                if proc.returncode != 0:  # show why it failed in the session log
                    with open(log_file.name, encoding="utf-8", errors="replace") as f:
                        print("".join(f.readlines()[-30:]), flush=True)
            event = {"run": f"{task}/{model}/seed{seed}", "returncode": proc.returncode}
            log_event(stage=stage["name"], **event, **({"gpu": slots[i]} if slots[i] is not None else {}))
            if proc.returncode != 0:
                completed_all = False
    if paused:
        return False
    log_event(stage=stage["name"], status="done" if completed_all else "done (some runs failed or waiting)")
    return completed_all


def stage_pip(stage, deadline) -> bool:
    code = run([PY, "-m", "pip", "install", "-q", *stage["packages"]])
    log_event(stage=stage["name"], status="done" if code == 0 else "failed", returncode=code)
    return code == 0


def stage_llm_sanity(stage, deadline) -> bool:
    """The released MentalGLM on its own test sets (scripts/run_llm_eval.py skips finished runs)."""
    cfg = load_config(resolve_path(stage["config"]))
    sc = cfg["sanity"]
    out = resolve_path(sc["output_root"])
    from run_llm_eval import sanity_run_name  # same folder names as the script writes
    missing = [f"{sanity_run_name(r)}/{t}" for r in sc["runs"] for t in sc["tasks"]
               if not (out / sanity_run_name(r) / t / "metrics.json").exists()]
    if not missing:
        log_event(stage=stage["name"], status="already done")
        return True
    if minutes_left(deadline) < stage.get("minutes", 60):
        log_event(stage=stage["name"], status="not started (time)")
        return False
    code = run([PY, "scripts/run_llm_eval.py", "--config", stage["config"]])
    log_event(stage=stage["name"], status="done" if code == 0 else "failed", returncode=code)
    return code == 0


def stage_llm_probe(stage, deadline) -> bool:
    """Time a few LoRA training steps and a small generation pass per task, to plan GPU time."""
    cfg = load_config(resolve_path(stage["config"]))
    probe_dir = resolve_path(cfg["output_root"]) / "_probe"
    ok = True
    for task in stage["tasks"]:
        for model in stage["models"]:
            if (probe_dir / f"{task}__{model}.json").exists():
                continue
            if minutes_left(deadline) < stage.get("minutes_per_task", 20):
                log_event(stage=stage["name"], status="paused (time)", next=f"{task}/{model}")
                return False
            code = run([PY, "scripts/run_llm_finetune.py", "--config", stage["config"], "--tasks", task,
                        "--models", model, "--probe-steps", str(stage.get("steps", 6))])
            log_event(stage=stage["name"], run=f"{task}/{model}", returncode=code)
            ok &= code == 0
    log_event(stage=stage["name"], status="done" if ok else "done (some probes failed)")
    return True  # a failed probe should not block the rest of the session


def stage_tfidf(stage, deadline) -> bool:
    return run([PY, "scripts/run_tfidf.py"]) == 0


def stage_summary(stage, deadline) -> bool:
    run([PY, "scripts/summarize_results.py"])
    for model_a, model_b in stage.get("compare", []):
        run([PY, "scripts/summarize_results.py", "--compare", model_a, model_b])
    return True


STAGES = {
    "preflight": stage_preflight,
    "dapt": stage_dapt,
    "finetune": stage_finetune,
    "tfidf": stage_tfidf,
    "summary": stage_summary,
    "pip": stage_pip,
    "llm_sanity": stage_llm_sanity,
    "llm_probe": stage_llm_probe,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--deadline", type=float, help="unix time by which to stop (already includes the margin)")
    ap.add_argument("--budget-hours", type=float, help="override the config budget (from now)")
    ap.add_argument("--no-limit", action="store_true", help="no time budget (e.g. a local GPU)")
    ap.add_argument("--dapt-set", nargs="*", default=[], help="config overrides for every DAPT run")
    ap.add_argument("--finetune-set", nargs="*", default=[], help="config overrides for every fine-tuning run")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    EXTRA["dapt"], EXTRA["finetune"] = args.dapt_set, args.finetune_set
    global ONE_RUN_PER_GPU
    ONE_RUN_PER_GPU = bool(cfg.get("finetune_one_run_per_gpu", False))

    if args.no_limit:
        deadline = time.time() + 10 * 365 * 24 * 3600
    elif args.deadline:
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
