import json

import numpy as np
import pytest

from matcha_bench.finetune.metrics import SINGLE, classification_metrics
from matcha_bench.results import bootstrap_macro_f1, check_run, compare, find_run, holm


def _sk_macro(labels, preds, k):
    return classification_metrics(labels, preds, SINGLE, "macro", [str(i) for i in range(k)], list(range(k)))["f1"]


def test_bootstrap_matches_sklearn_on_full_and_resampled_sets():
    rng = np.random.default_rng(1)
    k, n = 4, 300
    labels = rng.integers(0, k, n)
    a = np.where(rng.random(n) < 0.6, labels, rng.integers(-1, k, n))  # includes unparsed (-1) answers
    b = np.where(rng.random(n) < 0.4, labels, rng.integers(0, k, n))
    observed, boot = bootstrap_macro_f1(labels, [a, b], k, n_resamples=5, seed=3, chunk=2)
    assert observed == pytest.approx([_sk_macro(labels, a, k), _sk_macro(labels, b, k)], abs=1e-12)
    # replay the same draws to check one resample against sklearn
    draw = np.random.default_rng(3).integers(0, n, (2, n))[0]
    assert boot[0, 0] == pytest.approx(_sk_macro(labels[draw], a[draw], k), abs=1e-12)
    assert boot.shape == (5, 2)


def test_compare_finds_a_clear_difference_and_none_for_identical_models():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 2000)
    good = np.where(rng.random(2000) < 0.8, labels, (labels + 1) % 3)
    bad = np.where(rng.random(2000) < 0.5, labels, (labels + 1) % 3)
    obs, boot = bootstrap_macro_f1(labels, [good, bad, good], 3, n_resamples=500)
    res = compare(obs[[0]], boot[:, [0]], obs[[1]], boot[:, [1]])
    assert res["delta"] > 0.2 and res["ci_low"] > 0 and res["p_value"] < 0.01
    same = compare(obs[[0]], boot[:, [0]], obs[[2]], boot[:, [2]])
    assert same["delta"] == 0 and same["p_value"] == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _write_run(root, rel, f1, preds=None, labels=None, **extra):
    d = root / rel
    d.mkdir(parents=True)
    m = {"task": "t", "model_tag": "m", "problem_type": SINGLE, "label_names": ["a", "b"],
         "test": {"f1": f1, "average": "macro"}, **extra}
    (d / "metrics.json").write_text(json.dumps(m))
    if preds is not None:
        np.savez(d / "test_predictions.npz", labels=labels, preds=preds)


def test_find_run_prefers_full_test_and_saved_predictions(tmp_path):
    labels, preds = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
    f1 = _sk_macro(labels, preds, 2)
    model = {"path": "finetune/{task}/{tag}/seed{seed}"}
    _write_run(tmp_path / "r1", "finetune/t/m/seed1", 0.123, test_subset=True)  # superseded sample test
    _write_run(tmp_path / "r2", "finetune/t/m/seed1", f1)
    _write_run(tmp_path / "r3", "finetune/t/m/seed1", f1, preds, labels)
    run = find_run([tmp_path / r for r in ("r1", "r2", "r3")], "m", model, "t", 1)
    assert run.path == tmp_path / "r3/finetune/t/m/seed1" and run.copies == 2
    assert check_run(run) == pytest.approx(f1)


def test_find_run_rejects_disagreeing_copies_unless_matched(tmp_path):
    model = {"path": "zs/{task}/{tag}/seed{seed}"}
    _write_run(tmp_path / "r1", "zs/t/m/seed0", 0.2, hyperparameters={"max_new_tokens": 24})
    _write_run(tmp_path / "r2", "zs/t/m/seed0", 0.3, hyperparameters={"max_new_tokens": 96})
    roots = [tmp_path / "r1", tmp_path / "r2"]
    with pytest.raises(ValueError, match="disagree"):
        find_run(roots, "m", model, "t", 0)
    run = find_run(roots, "m", {**model, "match": {"hyperparameters.max_new_tokens": 96}}, "t", 0)
    assert run.f1 == 0.3


def test_check_run_catches_a_wrong_score(tmp_path):
    _write_run(tmp_path, "finetune/t/m/seed1", 0.99, np.array([0, 0]), np.array([0, 1]))
    run = find_run([tmp_path], "m", {"path": "finetune/{task}/{tag}/seed{seed}"}, "t", 1)
    with pytest.raises(ValueError, match="!= metrics.json"):
        check_run(run)
