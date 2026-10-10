from collections import Counter

import numpy as np
import pandas as pd

from matcha_bench.data.splits import assign_user_splits
from matcha_bench.finetune.metrics import MULTI, SINGLE, classification_metrics, predictions_from_logits
from matcha_bench.lexicon.mine import is_candidate, log_odds_dirichlet
from matcha_bench.stats import paired_bootstrap


def test_user_splits_are_disjoint_and_stratified():
    users = pd.DataFrame({"user_id": [f"u{i}" for i in range(1000)], "stratum": ["DEP"] * 700 + ["CONTROL"] * 300})
    split = assign_user_splits(users, "stratum", (0.7, 0.1, 0.2), seed=1)
    assert set(split) == {"train", "dev", "test"}
    shares = split.value_counts(normalize=True)
    assert abs(shares["test"] - 0.2) < 0.01 and abs(shares["dev"] - 0.1) < 0.01
    test_ctrl = (users.loc[split == "test", "stratum"] == "CONTROL").mean()
    assert abs(test_ctrl - 0.3) < 0.02


def test_metrics_binary_macro_micro():
    y = np.array([0, 1, 1, 0])
    p = np.array([0, 1, 0, 0])
    m = classification_metrics(y, p, SINGLE, "binary", ["neg", "pos"])
    assert m["precision"] == 1.0 and m["recall"] == 0.5
    ym = np.array([[1, 0], [0, 1]])
    pm = predictions_from_logits(np.array([[2.0, -1.0], [-1.0, -0.5]]), MULTI)
    mm = classification_metrics(ym, pm, MULTI, "micro", ["a", "b"])
    assert mm["precision"] == 1.0 and mm["recall"] == 0.5


def test_log_odds_ranks_overrepresented_word_first():
    target = Counter({"失眠": 100, "今天": 100, "吃饭": 10})
    control = Counter({"失眠": 2, "今天": 100, "吃饭": 50})
    prior = target + control
    scores = log_odds_dirichlet(target, control, prior).sort_values("z", ascending=False)
    assert scores.iloc[0]["word"] == "失眠"
    assert scores.iloc[-1]["word"] == "吃饭"


def test_apply_word_annotations_keeps_listed_words_for_every_disorder():
    from matcha_bench.lexicon.io import apply_word_annotations

    cands = pd.DataFrame({"disorder": ["DEP", "GAD", "GAD"], "word": ["失眠", "失眠", "红包"]})
    ann = pd.DataFrame({"word": ["失眠", "错字"], "category": ["somatic", "emotion"]})
    out, unmatched = apply_word_annotations(cands, ann)
    assert out["keep"].tolist() == [1, 1, 0]
    assert out.loc[out["word"] == "红包", "category"].item() == "none"
    assert unmatched == ["错字"]


def test_candidate_filter():
    assert is_candidate("失眠") and is_candidate("emo")
    assert not is_candidate("的") and not is_candidate("2023") and not is_candidate("哈哈")


def test_paired_bootstrap_detects_clear_win():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(rng.random(400) < 0.9, y, 1 - y)
    bad = np.where(rng.random(400) < 0.6, y, 1 - y)
    res = paired_bootstrap(y, good, bad, lambda a, b: (a == b).mean(), n_resamples=200)
    assert res["delta"] > 0 and res["p_value"] < 0.05
