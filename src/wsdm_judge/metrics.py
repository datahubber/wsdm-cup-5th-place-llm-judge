"""Accuracy with uncertainty, pairwise heuristic baselines, and an offline tau sweep.

The competition metric is plain accuracy on A/B. A single leaderboard score over n rows has a
binomial standard error of about sqrt(p(1-p)/n). Two post-processing variants of the same model
differ only on the rows they flip, so they are compared with an exact sign test on those rows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wsdm_judge.postprocess import gated_pick_a

# Size of the public leaderboard split. Every public score of the team (130 distinct values) is a
# multiple of 1/9997 up to the 6-decimal rounding, so one row is worth 0.0001 accuracy.
PUBLIC_LB_ROWS = 9997


def accuracy(pred: Sequence[bool] | np.ndarray, gold: Sequence[bool] | np.ndarray) -> float:
    p, g = np.asarray(pred, dtype=bool), np.asarray(gold, dtype=bool)
    if p.shape != g.shape or p.size == 0:
        raise ValueError("pred and gold must be non-empty and the same shape")
    return float((p == g).mean())


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (95 % by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def sign_test_pvalue(wins: int, losses: int) -> float:
    """Exact two-sided sign test: is a changed decision more often right than wrong?"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def lb_rows(delta: float, n: int = PUBLIC_LB_ROWS) -> int:
    """Convert an accuracy difference on the leaderboard into a net number of rows."""
    return round(delta * n)


@dataclass(frozen=True)
class RuleResult:
    """A pairwise rule "the response with the larger feature wins" (ties abstain)."""

    name: str
    n_rows: int
    coverage: float  # share of rows where the rule makes a call
    n_decided: int
    accuracy: float  # accuracy on the rows where it makes a call
    ci_low: float
    ci_high: float


def pairwise_rule(
    name: str,
    feature_a: Sequence[float] | np.ndarray,
    feature_b: Sequence[float] | np.ndarray,
    a_won: Sequence[bool] | np.ndarray,
    mask: Sequence[bool] | np.ndarray | None = None,
) -> RuleResult:
    fa, fb = np.asarray(feature_a, dtype=float), np.asarray(feature_b, dtype=float)
    y = np.asarray(a_won, dtype=bool)
    rows = np.ones_like(y) if mask is None else np.asarray(mask, dtype=bool)
    decided = rows & (fa != fb)
    n_rows, n_dec = int(rows.sum()), int(decided.sum())
    if n_dec == 0:
        return RuleResult(name, n_rows, 0.0, 0, float("nan"), float("nan"), float("nan"))
    correct = int(((fa > fb) == y)[decided].sum())
    lo, hi = wilson_interval(correct, n_dec)
    return RuleResult(name, n_rows, n_dec / max(n_rows, 1), n_dec, correct / n_dec, lo, hi)


@dataclass(frozen=True)
class SweepRow:
    tau: float
    in_band: float  # share of rows with |margin| < tau
    changed: int  # rows whose decision differs from the plain argmax
    wins: int  # changed rows that became correct
    losses: int  # changed rows that became wrong
    accuracy: float
    p_value: float  # sign test on (wins, losses)


def sweep_tau(
    lp_a: Sequence[float] | np.ndarray,
    lp_b: Sequence[float] | np.ndarray,
    count_a: Sequence[int] | np.ndarray,
    count_b: Sequence[int] | np.ndarray,
    a_won: Sequence[bool] | np.ndarray,
    taus: Sequence[float],
) -> list[SweepRow]:
    """Accuracy of the gated rule for each tau, against the plain argmax (tau = 0).

    Meant for out-of-fold log-probabilities with labels, so tau can be chosen on validation data
    instead of on the leaderboard.
    """
    if not len(lp_a) == len(lp_b) == len(count_a) == len(count_b) == len(a_won):
        raise ValueError("all inputs must have the same length")
    margin = np.asarray(lp_a, dtype=float) - np.asarray(lp_b, dtype=float)
    ca, cb = np.asarray(count_a, dtype=int), np.asarray(count_b, dtype=int)
    y = np.asarray(a_won, dtype=bool)
    base = margin > 0
    rows: list[SweepRow] = []
    for tau in taus:
        pick_a = gated_pick_a(margin, ca, cb, tau)
        changed = pick_a != base
        wins = int((changed & (pick_a == y)).sum())
        losses = int((changed & (pick_a != y)).sum())
        rows.append(
            SweepRow(
                tau=float(tau),
                in_band=float((np.abs(margin) < tau).mean()),
                changed=int(changed.sum()),
                wins=wins,
                losses=losses,
                accuracy=accuracy(pick_a, y),
                p_value=sign_test_pvalue(wins, losses),
            )
        )
    return rows
