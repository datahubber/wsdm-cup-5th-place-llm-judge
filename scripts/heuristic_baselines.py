"""How much signal do the post-processor's fallback features carry on their own?

Each rule says "the response with the larger feature wins" and abstains on ties. The features are
the ones the fallback was tried with on the leaderboard (docs/leaderboard_log.md). Runs on the
labelled competition train set on CPU and writes a Markdown report with 95 % Wilson intervals.

    python scripts/heuristic_baselines.py --data data/train.parquet \
        --out results/heuristic_baselines.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from wsdm_judge.data import a_won, read_table
from wsdm_judge.metrics import RuleResult, pairwise_rule, wilson_interval
from wsdm_judge.postprocess import count_numbers

LENGTH_BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, float("inf")))


def _avg_word_len(text: str) -> float:
    words = text.split()
    return float(np.mean([len(w) for w in words])) if words else 0.0


def features(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def both(fn: Callable[[str], float]) -> tuple[np.ndarray, np.ndarray]:
        return df["response_a"].map(fn).to_numpy(float), df["response_b"].map(fn).to_numpy(float)

    return {
        "more digit runs (the fallback used)": both(count_numbers),
        "more characters": both(len),
        "more words": both(lambda t: len(t.split())),
        "longer average word": both(_avg_word_len),
    }


def _rule_row(r: RuleResult) -> str:
    return (
        f"| {r.name} | {r.n_decided:,} / {r.n_rows:,} ({r.coverage:.1%}) | "
        f"{r.accuracy:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |"
    )


def report(df: pd.DataFrame, top_languages: int = 6) -> str:
    y = a_won(df)
    feats = features(df)
    num_a, num_b = feats["more digit runs (the fallback used)"]
    len_a, len_b = feats["more characters"]
    n = len(df)
    header = "| Rule | Rows decided | Accuracy when decided | 95 % CI |\n|---|---|---|---|"

    out = [
        "# Fallback features on their own (competition train set)",
        "",
        f"Rows: {n:,}. Distinct prompts: {df['prompt'].nunique():,}. "
        f"Share of rows where B won: {1 - y.mean():.4f}.",
        "Rows that share a prompt are not independent, so the intervals are somewhat too narrow.",
        "",
        "## 1. Each rule alone",
        "",
        header,
    ]
    out += [_rule_row(pairwise_rule(name, fa, fb, y)) for name, (fa, fb) in feats.items()]

    # Is "more numbers" more than a length proxy? Look where the two rules disagree, and inside
    # bins of similar length.
    disagree = (num_a != num_b) & (len_a != len_b) & ((num_a > num_b) != (len_a > len_b))
    k = int(((num_a > num_b) == y)[disagree].sum())
    m = int(disagree.sum())
    out += ["", "## 2. Digit runs versus length", ""]
    if m:
        lo, hi = wilson_interval(k, m)
        out.append(
            f"Rows where the two rules pick different responses: {m:,}. The digit-run rule is "
            f"right on {k / m:.4f} of them (95 % CI [{lo:.4f}, {hi:.4f}])."
        )
    else:
        out.append("The two rules never pick different responses on this data.")
    out += [
        "",
        "Within bins of |log(chars A / chars B)|:",
        "",
        "| Length ratio bin | Digit-run rule: decided, accuracy | Length rule: decided, accuracy |",
        "|---|---|---|",
    ]
    ratio = np.abs(np.log((len_a + 1) / (len_b + 1)))
    for lo_b, hi_b in LENGTH_BINS:
        in_bin = (ratio >= lo_b) & (ratio < hi_b)
        rn = pairwise_rule("n", num_a, num_b, y, mask=in_bin)
        rl = pairwise_rule("l", len_a, len_b, y, mask=in_bin)
        label = f"[{lo_b:g}, {hi_b:g})" if np.isfinite(hi_b) else f">= {lo_b:g}"
        out.append(
            f"| {label} | {rn.n_decided:,}, {rn.accuracy:.4f} "
            f"[{rn.ci_low:.4f}, {rn.ci_high:.4f}] | {rl.n_decided:,}, {rl.accuracy:.4f} "
            f"[{rl.ci_low:.4f}, {rl.ci_high:.4f}] |"
        )

    if "language" in df.columns:
        out += [
            "",
            f"## 3. Digit-run rule by language (top {top_languages} by rows)",
            "",
            "| Language | Rows | Digit-run rule: decided, accuracy [95 % CI] |",
            "|---|---|---|",
        ]
        for lang in df["language"].value_counts().head(top_languages).index:
            mask = (df["language"] == lang).to_numpy()
            r = pairwise_rule(lang, num_a, num_b, y, mask=mask)
            out.append(
                f"| {lang} | {int(mask.sum()):,} | {r.n_decided:,}, {r.accuracy:.4f} "
                f"[{r.ci_low:.4f}, {r.ci_high:.4f}] |"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="labelled pairs: .parquet, .csv or .jsonl")
    ap.add_argument("--out", help="write the Markdown report here as well")
    ap.add_argument("--top-languages", type=int, default=6)
    args = ap.parse_args(argv)

    text = report(read_table(args.data), args.top_languages)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
