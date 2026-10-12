"""The fine-tuning scheduler in scripts/run_pipeline.py, with fake runs instead of training."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_pipeline

    events, launched = [], []
    monkeypatch.setattr(run_pipeline, "log_event", lambda **e: events.append(e))
    monkeypatch.setattr(run_pipeline, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(run_pipeline, "gpu_slots", lambda: ["0", "1"])

    def fake_start(task, model, seed, device):
        active = sum(p.poll() is None for *_, p in launched)
        code = 3 if model == "broken" else 0
        proc = subprocess.Popen([sys.executable, "-c", f"import time, sys; time.sleep(0.3); sys.exit({code})"])
        launched.append((task, model, seed, device, active + 1, proc))
        return proc, None

    monkeypatch.setattr(run_pipeline, "start_finetune", fake_start)

    task_root = tmp_path / "tasks"
    (task_root / "t1").mkdir(parents=True)
    (task_root / "t1" / "meta.json").write_text("{}", encoding="utf-8")
    ft = {"task_root": str(task_root), "output_root": str(tmp_path / "out"), "seeds": [1, 2],
          "models": {"a": "hub/a", "b": "hub/b", "broken": "hub/broken"}}
    (tmp_path / "ft.yaml").write_text(yaml.safe_dump(ft), encoding="utf-8")

    def stage(models):
        return {"name": "finetune-test", "config": str(tmp_path / "ft.yaml"), "tasks": ["t1"],
                "models": models, "minutes_per_run": {"default": 20}}

    return run_pipeline, stage, events, launched


def test_runs_one_job_per_gpu_in_parallel(pipeline):
    rp, stage, events, launched = pipeline
    assert rp.stage_finetune(stage(["a", "b"]), deadline=time.time() + 3600) is True
    assert len(launched) == 4
    assert max(active for *_, active, _ in launched) == 2  # two GPUs busy, never more
    runs = [e for e in events if "run" in e]
    assert sorted(e["run"] for e in runs) == ["t1/a/seed1", "t1/a/seed2", "t1/b/seed1", "t1/b/seed2"]
    assert {e["gpu"] for e in runs} == {"0", "1"}
    assert events[-1]["status"] == "done"


def test_pauses_before_a_run_that_does_not_fit(pipeline):
    rp, stage, events, launched = pipeline
    assert rp.stage_finetune(stage(["a"]), deadline=time.time() + 60) is False  # 1 min < 20 reserved
    assert launched == []
    assert events[-1]["status"] == "paused (time)"


def test_a_failed_run_is_reported_but_others_continue(pipeline):
    rp, stage, events, launched = pipeline
    assert rp.stage_finetune(stage(["broken", "a"]), deadline=time.time() + 3600) is False
    assert len(launched) == 4
    codes = {e["run"]: e["returncode"] for e in events if "run" in e}
    assert codes["t1/broken/seed1"] == 3 and codes["t1/a/seed1"] == 0
    assert events[-1]["status"] == "done (some runs failed or waiting)"
