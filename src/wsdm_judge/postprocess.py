"""Uncertainty-gated tie-break for a pairwise judge.

The judge's preference is the margin m = logp("A") - logp("B"). Restricted to the two labels,
P(A) = sigmoid(m), so |m| < tau means the judge is within sigmoid(tau) - 0.5 of a coin flip
(tau = 0.01 -> 0.25 percentage points). Inside that band the rule defers to a content signal:
the response with more digit runs ("2025", "3.5" counts as two) wins. Outside the band, or when
both responses have the same count, the judge's argmax stands.

The same gate works on any margin. The earlier Gemma-2 pipeline used m = P(A) - P(B) from a
3-class softmax (see docs/leaderboard_log.md, section "Sweep A").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Winner = Literal["A", "B"]
# judge          : |margin| >= tau, the judge's argmax
# numbers        : |margin| < tau and the digit-run counts differ
# judge_in_band  : |margin| < tau but equal counts, so the judge's argmax
# one_label      : only one of "A"/"B" was in the top-k, so that one
# no_label       : neither label was in the top-k, default "A"
Reason = Literal["judge", "numbers", "judge_in_band", "one_label", "no_label"]

_NUMBER = re.compile(r"\d+")  # \d matches Unicode digits, e.g. Arabic-Indic and full-width ones


@dataclass(frozen=True)
class Decision:
    winner: Winner
    reason: Reason


def count_numbers(text: str) -> int:
    """Number of maximal digit runs in `text`."""
    return len(_NUMBER.findall(text))


def prob_band(tau: float) -> float:
    """Half-width of the band around P(A) = 0.5 that a logprob gate `tau` covers."""
    return 1.0 / (1.0 + math.exp(-tau)) - 0.5


def gated_choice(margin: float, count_a: int, count_b: int, tau: float) -> Decision:
    """Core rule on a precomputed margin (positive favours A). Ties at margin == 0 go to B."""
    if abs(margin) < tau:
        if count_a != count_b:
            return Decision("A" if count_a > count_b else "B", "numbers")
        return Decision("A" if margin > 0 else "B", "judge_in_band")
    return Decision("A" if margin > 0 else "B", "judge")


def gated_pick_a(
    margin: Sequence[float] | np.ndarray,
    count_a: Sequence[int] | np.ndarray,
    count_b: Sequence[int] | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorised `gated_choice` over many rows: True where the rule picks A."""
    m = np.asarray(margin, dtype=float)
    ca, cb = np.asarray(count_a), np.asarray(count_b)
    return np.where((np.abs(m) < tau) & (ca != cb), ca > cb, m > 0)


def decide(
    lp_a: float | None, lp_b: float | None, text_a: str, text_b: str, tau: float
) -> Decision:
    """Final A/B choice for one pair from the judge's label log-probabilities."""
    if lp_a is None and lp_b is None:
        return Decision("A", "no_label")
    if lp_a is None or lp_b is None:
        return Decision("A" if lp_b is None else "B", "one_label")
    return gated_choice(lp_a - lp_b, count_numbers(text_a), count_numbers(text_b), tau)


def decide_all(
    lp_a: Sequence[float | None],
    lp_b: Sequence[float | None],
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    tau: float,
) -> list[Decision]:
    if not len(lp_a) == len(lp_b) == len(texts_a) == len(texts_b):
        raise ValueError("all inputs must have the same length")
    return [decide(*row, tau=tau) for row in zip(lp_a, lp_b, texts_a, texts_b, strict=True)]


def reason_counts(decisions: Iterable[Decision]) -> dict[str, int]:
    return dict(Counter(d.reason for d in decisions))


def to_label(winner: Winner) -> str:
    """Competition label: "model_a" or "model_b"."""
    return "model_a" if winner == "A" else "model_b"
