"""Re-apply the gate to saved judge log-probabilities, without re-running the model (CPU).

If --data has labels (a `winner` column), it also prints a tau sweep: for each tau, the rows whose
decision changes versus the plain argmax, how many of those became right or wrong, and an exact
sign test. With out-of-fold log-probabilities this picks tau on validation data instead of on the
leaderboard, which the competition run did not do.

    python scripts/apply_gate.py --data test.parquet --logprobs logprobs.csv --tau 0.005
    python scripts/apply_gate.py --data oof.parquet --logprobs oof.csv --sweep 0,0.005,0.01,0.05
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from wsdm_judge.data import a_won, read_table
from wsdm_judge.metrics import sweep_tau
from wsdm_judge.postprocess import count_numbers, decide_all, reason_counts, to_label


def _optional(x: Any) -> float | None:
    """A pandas cell (float, NaN or None) as a float, or None when missing."""
    return None if pd.isna(x) else float(x)


def sweep_table(df: pd.DataFrame, taus: list[float]) -> str:
    """Markdown sweep table over the rows where both label log-probabilities exist."""
    ok = df["lp_a"].notna() & df["lp_b"].notna()
    d = df[ok]
    rows = sweep_tau(
        d["lp_a"].to_numpy(float),
        d["lp_b"].to_numpy(float),
        d["response_a"].map(count_numbers).to_numpy(),
        d["response_b"].map(count_numbers).to_numpy(),
        a_won(d),
        taus,
    )
    lines = [
        f"Rows with both label log-probabilities: {int(ok.sum())} of {len(df)}",
        "",
        "| tau | in band | changed | became right | became wrong | accuracy | sign-test p |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.tau:g} | {r.in_band:.2%} | {r.changed} | {r.wins} | {r.losses} | "
        f"{r.accuracy:.4f} | {r.p_value:.3g} |"
        for r in rows
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="pairs with id, response_a, response_b")
    ap.add_argument("--logprobs", required=True, help="csv with id, lp_a, lp_b from predict.py")
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--out", help="write a submission csv here")
    ap.add_argument("--sweep", help="comma-separated taus; needs a `winner` column in --data")
    args = ap.parse_args(argv)

    df = read_table(args.data).merge(read_table(args.logprobs), on="id", how="left")
    if df["lp_a"].isna().all() and df["lp_b"].isna().all():
        raise SystemExit("no ids matched between --data and --logprobs")

    decisions = decide_all(
        [_optional(x) for x in df["lp_a"]],
        [_optional(x) for x in df["lp_b"]],
        df["response_a"].tolist(),
        df["response_b"].tolist(),
        args.tau,
    )
    print(f"tau={args.tau:g}; decisions by reason: {reason_counts(decisions)}")
    pick_a = np.array([d.winner == "A" for d in decisions])
    if "winner" in df.columns:
        print(f"accuracy at tau={args.tau:g}: {(pick_a == a_won(df)).mean():.4f} ({len(df)} rows)")
    if args.out:
        labels = [to_label(d.winner) for d in decisions]
        pd.DataFrame({"id": df["id"], "winner": labels}).to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    if args.sweep:
        if "winner" not in df.columns:
            raise SystemExit("--sweep needs labels (a `winner` column) in --data")
        print(sweep_table(df, [float(t) for t in args.sweep.split(",")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
