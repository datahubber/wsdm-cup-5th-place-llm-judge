"""Draw the README figure from committed numbers only (no data, no model).

Left and middle: net public-LB rows gained by the gate against the no-gate reference, per tau,
from `results/gate_sweeps.csv` (scores from the Kaggle submission list; see
docs/leaderboard_log.md). Right: the digit-run rule against the length rule on the train set,
per length-ratio bin, from `results/heuristic_baselines.md`.

    pip install -e ".[plot]"
    python scripts/plot_results.py --out assets/results.png
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from wsdm_judge.metrics import PUBLIC_LB_ROWS, lb_rows

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Binomial standard error of one public score near 0.70, in rows: sqrt(0.7 * 0.3 * 9997) = 45.8.
SINGLE_SCORE_SE_ROWS = 46

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


@dataclass(frozen=True)
class BinRow:
    """One length-ratio bin: accuracy and 95 % CI of the digit-run and the length rule."""

    label: str
    digit_acc: float
    digit_ci: tuple[float, float]
    length_acc: float
    length_ci: tuple[float, float]


def load_sweeps(path: str | Path) -> pd.DataFrame:
    """Sweep table plus a `rows` column: net public-LB rows against the reference."""
    df = pd.read_csv(path)
    deltas = df["public_lb"] - df["reference_lb"]
    df["rows"] = [lb_rows(d, PUBLIC_LB_ROWS) for d in deltas]
    return df


_CELL = re.compile(r"[\d,]+, ([\d.]+) \[([\d.]+), ([\d.]+)\]")


def load_length_bins(path: str | Path) -> list[BinRow]:
    """Parse the per-length-ratio-bin table of results/heuristic_baselines.md."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[0].startswith(("[", ">=")):
            continue
        digit, length = _CELL.fullmatch(cells[1]), _CELL.fullmatch(cells[2])
        if digit is None or length is None:
            continue
        rows.append(
            BinRow(
                label=cells[0],
                digit_acc=float(digit[1]),
                digit_ci=(float(digit[2]), float(digit[3])),
                length_acc=float(length[1]),
                length_ci=(float(length[2]), float(length[3])),
            )
        )
    if not rows:
        raise ValueError(f"no length-ratio bins found in {path}")
    return rows


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def _gate_panel(ax: Axes, df: pd.DataFrame, title: str, xlabel: str) -> None:
    se = SINGLE_SCORE_SE_ROWS
    ax.axhspan(-se, se, color=GRID, alpha=0.55, linewidth=0, label=f"±{se} rows: 1 SE of one score")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for i, (name, g) in enumerate(df.groupby("series", sort=False)):
        g = g.sort_values("tau")
        style = (
            {"color": BLUE, "linewidth": 2, "marker": "o", "markersize": 5}
            if i == 0
            else {
                "linestyle": "none",
                "marker": "D",
                "markersize": 7,
                "markerfacecolor": SURFACE,
                "markeredgecolor": BLUE,
                "markeredgewidth": 2,
            }
        )
        ax.plot(g["tau"], g["rows"], label=str(name), **style)
    best = df.loc[df["rows"].idxmax()]
    ax.annotate(
        f"{int(best['rows']):+d}",
        (best["tau"], best["rows"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color=INK,
    )
    _style(ax, title, xlabel, "Public-LB rows vs. no gate (of 9,997)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", labelcolor=MUTED)


def _length_panel(ax: Axes, bins: list[BinRow]) -> None:
    xs = list(range(len(bins)))
    series = (
        (-0.12, BLUE, "more digit runs wins", [(b.digit_acc, b.digit_ci) for b in bins]),
        (0.12, ORANGE, "longer response wins", [(b.length_acc, b.length_ci) for b in bins]),
    )
    for offset, color, name, points in series:
        ys = [acc for acc, _ in points]
        err = [[acc - lo for acc, (lo, _) in points], [hi - acc for acc, (_, hi) in points]]
        ax.errorbar(
            [x + offset for x in xs],
            ys,
            yerr=err,
            fmt="o",
            color=color,
            markersize=6,
            elinewidth=2,
            capsize=0,
            label=name,
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.set_xticks(xs, [b.label for b in bins])
    _style(
        ax,
        "Digit runs track length (48,439 train pairs)",
        "|log(chars A / chars B)|, binned",
        "Accuracy where the rule decides (95 % CI)",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=MUTED)


def plot(sweeps: pd.DataFrame, bins: list[BinRow], out: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    fig.patch.set_facecolor(SURFACE)
    gemma = sweeps[sweeps["panel"] == "gemma"]
    qwen = sweeps[sweeps["panel"] == "qwen"]
    _gate_panel(axes[0], gemma, "Gate on a Gemma-2 baseline", "τ on P(A) − P(B)")
    _gate_panel(axes[1], qwen, "Gate on the fine-tuned Qwen judge", "τ on log-prob margin")
    lo = min(int(sweeps["rows"].min()), -SINGLE_SCORE_SE_ROWS) - 15
    hi = max(int(sweeps["rows"].max()), SINGLE_SCORE_SE_ROWS) + 20
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    _length_panel(axes[2], bins)
    fig.tight_layout(w_pad=2.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweeps", default="results/gate_sweeps.csv")
    ap.add_argument("--baselines", default="results/heuristic_baselines.md")
    ap.add_argument("--out", default="assets/results.png")
    args = ap.parse_args(argv)
    plot(load_sweeps(args.sweeps), load_length_bins(args.baselines), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
