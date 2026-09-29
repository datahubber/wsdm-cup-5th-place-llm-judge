from __future__ import annotations

import numpy as np
import pytest

from wsdm_judge.postprocess import (
    Decision,
    count_numbers,
    decide,
    decide_all,
    gated_choice,
    gated_pick_a,
    prob_band,
    reason_counts,
    to_label,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no digits here", 0),
        ("version 3.5", 2),
        ("in 2025", 1),
        ("２０２５年", 1),  # full-width digits
        ("٣ و ٤", 2),  # Arabic-Indic digits
    ],
)
def test_count_numbers(text, expected):
    assert count_numbers(text) == expected


def test_confident_margin_keeps_the_judge():
    assert gated_choice(0.5, 0, 10, tau=0.01) == Decision("A", "judge")
    assert gated_choice(-0.5, 10, 0, tau=0.01) == Decision("B", "judge")


def test_near_tie_defers_to_digit_runs():
    assert gated_choice(0.004, 1, 4, tau=0.01) == Decision("B", "numbers")
    assert gated_choice(-0.004, 4, 1, tau=0.01) == Decision("A", "numbers")


def test_near_tie_with_equal_counts_keeps_the_judge():
    assert gated_choice(0.004, 2, 2, tau=0.01) == Decision("A", "judge_in_band")
    assert gated_choice(0.0, 2, 2, tau=0.01) == Decision("B", "judge_in_band")


def test_gate_is_strict_and_tau_zero_is_argmax():
    assert gated_choice(0.01, 0, 5, tau=0.01).reason == "judge"
    assert gated_choice(1e-9, 0, 5, tau=0.0) == Decision("A", "judge")


@pytest.mark.parametrize("tau", [0.0, 0.01, 0.05, 1.0])
def test_vectorised_gate_matches_the_scalar_rule(tau):
    rng = np.random.default_rng(1)
    m = np.concatenate([rng.normal(0, 0.05, 1000), [0.0, tau, -tau]])
    ca, cb = rng.integers(0, 4, m.size), rng.integers(0, 4, m.size)
    want = [gated_choice(x, a, b, tau).winner == "A" for x, a, b in zip(m, ca, cb, strict=True)]
    assert gated_pick_a(m, ca, cb, tau).tolist() == want


def test_missing_label_tokens():
    assert decide(None, None, "1", "2", tau=0.01) == Decision("A", "no_label")
    assert decide(-0.3, None, "", "", tau=0.01) == Decision("A", "one_label")
    assert decide(None, -0.3, "", "", tau=0.01) == Decision("B", "one_label")


def test_decide_all_and_reason_counts():
    decisions = decide_all(
        [-0.1, -0.69, None], [-2.0, -0.695, None], ["", "", ""], ["", "7", ""], 0.01
    )
    assert [d.winner for d in decisions] == ["A", "B", "A"]
    assert reason_counts(decisions) == {"judge": 1, "numbers": 1, "no_label": 1}
    with pytest.raises(ValueError):
        decide_all([0.0], [0.0, 1.0], ["a"], ["b"], 0.01)


def test_prob_band_matches_sigmoid():
    assert prob_band(0.01) == pytest.approx(0.0025, abs=1e-6)
    assert prob_band(0.05) == pytest.approx(0.0125, abs=1e-5)


def test_to_label():
    assert (to_label("A"), to_label("B")) == ("model_a", "model_b")
