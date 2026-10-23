"""Every pipeline config points at files that exist and names known models and tasks."""

from pathlib import Path

import pytest

from matcha_bench.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = sorted(ROOT.glob("configs/pipeline_*.yaml")) + sorted(ROOT.glob("configs/kaggle/pipeline_*.yaml"))


def known_tasks() -> set:
    tasks = load_config(ROOT / "configs" / "tasks.yaml")
    return set(tasks["matcha"]) | set(tasks["mentalbert"]) | {"reddit_posts"}


@pytest.mark.parametrize("path", PIPELINES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_pipeline_references_exist(path):
    pipe = load_config(path)
    tasks = known_tasks()
    for stage in pipe["stages"]:
        for key in ("config", "script"):
            if key in stage:
                assert (ROOT / stage[key]).exists(), f"{stage['name']}: {stage[key]}"
        for key in ("prepare", "before"):
            if key in stage:
                assert (ROOT / stage[key][0]).exists(), f"{stage['name']}: {stage[key][0]}"
        if stage["type"] in ("finetune", "llm_probe"):
            models = load_config(ROOT / stage["config"])["models"]
            assert set(stage["models"]) <= set(models), f"{stage['name']}: unknown model"
            assert set(stage["tasks"]) <= tasks, f"{stage['name']}: unknown task"


def test_results_config_models_cover_the_main_table():
    cfg = load_config(ROOT / "configs" / "results.yaml")
    assert set(cfg["main"]["models"]) <= set(cfg["models"])
    assert set(cfg["main"]["tasks"]) <= known_tasks()
    for spec in cfg["sanity"]:
        assert set(spec["models"]) <= set(cfg["models"])
        assert all(rep["ours"] in spec["models"] for rep in spec["reported"])
