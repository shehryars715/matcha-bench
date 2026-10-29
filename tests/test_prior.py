"""Class-prior correction: the shift, and tau picked on dev."""

import numpy as np

from matcha_bench.prior import log_shift, macro_f1, tune_tau


def test_shift_is_dev_mix_over_the_mix_the_model_was_trained_toward():
    dev = np.array([0] * 6 + [1] * 3 + [2] * 1)
    np.testing.assert_allclose(log_shift(dev, 3), np.log([0.6, 0.3, 0.1]) - np.log(1 / 3))
    np.testing.assert_allclose(log_shift(dev, 3, train_mix=[0.6, 0.3, 0.1]), 0, atol=1e-12)


def test_tau_undoes_a_known_prior_shift():
    # a model calibrated to a uniform mix, tested where class 0 is 9x as common
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    evidence = rng.normal(np.where(y == 1, 1.0, -1.0), 1.5)  # log-likelihood ratio of class 1
    scores = np.stack([-evidence / 2, evidence / 2], 1)
    dev, test = slice(0, 2000), slice(2000, None)
    shift = log_shift(y[dev], 2)
    tau, f1 = tune_tau(scores[dev], y[dev], shift, np.arange(0, 2.01, 0.1))
    assert tau > 0 and f1 > macro_f1(y[dev], scores[dev])
    assert macro_f1(y[test], scores[test] + tau * shift) > macro_f1(y[test], scores[test])
    # no shift to undo: tau stays at the first grid value
    assert tune_tau(scores[dev], y[dev], np.zeros(2), [0.0, 1.0])[0] == 0.0
