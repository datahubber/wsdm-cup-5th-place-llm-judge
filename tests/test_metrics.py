from __future__ import annotations

import numpy as np
import pytest

from wsdm_judge.metrics import (
    PUBLIC_LB_ROWS,
    accuracy,
    lb_rows,
    pairwise_rule,
    sign_test_pvalue,
    sweep_tau,
    wilson_interval,
)


def test_accuracy():
    assert accuracy([True, False, True], [True, True, True]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        accuracy([], [])


def test_wilson_interval_is_symmetric_at_half():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(1 - hi)
    assert lo == pytest.approx(0.4038, abs=1e-3)


def test_sign_test():
    assert sign_test_pvalue(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_pvalue(5, 5) == 1.0
    assert sign_test_pvalue(0, 0) == 1.0


def test_lb_rows():
    assert PUBLIC_LB_ROWS == 9997
    assert lb_rows(0.665899 - 0.661298) == 46


def test_pairwise_rule_abstains_on_ties():
    r = pairwise_rule("x", [3, 1, 2, 5], [1, 1, 4, 0], [True, True, False, False])
    assert (r.n_rows, r.n_decided) == (4, 3)
    assert r.accuracy == pytest.approx(2 / 3)
    assert r.ci_low < r.accuracy < r.ci_high


# The six fixture pairs with both log-probabilities (see tests/fixtures).
LP_A = [-0.1, -2.0, -0.693, -0.700, -0.690, -0.694]
LP_B = [-2.4, -0.2, -0.697, -0.695, -0.698, -0.690]
NUM_A = [0, 3, 1, 5, 0, 2]
NUM_B = [0, 0, 4, 2, 2, 2]
A_WON = [True, False, False, True, True, False]


def test_sweep_tau_counts_changed_rows():
    base, gated = sweep_tau(LP_A, LP_B, NUM_A, NUM_B, A_WON, [0.0, 0.01])
    assert (base.changed, base.accuracy) == (0, pytest.approx(4 / 6))
    assert (gated.changed, gated.wins, gated.losses) == (3, 2, 1)
    assert gated.accuracy == pytest.approx(5 / 6)
    assert gated.in_band == pytest.approx(4 / 6)


def test_sweep_tau_band_grows_with_tau():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.05, 500)
    rows = sweep_tau(
        m,
        np.zeros(500),
        rng.integers(0, 5, 500),
        rng.integers(0, 5, 500),
        rng.random(500) < 0.5,
        [0.0, 0.01, 0.05, 0.2],
    )
    bands = [r.in_band for r in rows]
    assert bands == sorted(bands) and bands[0] == 0.0 and bands[-1] == 1.0
