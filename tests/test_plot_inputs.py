"""The README figure is drawn from committed numbers; check they match docs/leaderboard_log.md."""

from __future__ import annotations

import plot_results
import pytest


def test_gate_sweeps_reproduce_the_logged_row_deltas(repo_root):
    df = plot_results.load_sweeps(repo_root / "results" / "gate_sweeps.csv")
    gemma = df[df["panel"] == "gemma"]
    assert gemma["rows"].max() == 46 and gemma["rows"].min() == -101
    sweep_b = df[df["series"].str.startswith("swap")].sort_values("tau")
    assert sweep_b["rows"].tolist() == [1, -2, -4, -8, -18]
    final = df[df["series"].str.startswith("final")]
    assert final["rows"].tolist() == [1]


def test_length_bins_parse_the_committed_report(repo_root):
    bins = plot_results.load_length_bins(repo_root / "results" / "heuristic_baselines.md")
    assert [b.label for b in bins] == ["[0, 0.1)", "[0.1, 0.3)", "[0.3, 0.7)", ">= 0.7"]
    assert (bins[0].digit_acc, bins[0].length_acc) == (0.5477, 0.5258)
    assert bins[2].length_ci == (0.5963, 0.6118)


def test_length_bins_reject_a_report_without_the_table(tmp_path):
    empty = tmp_path / "report.md"
    empty.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no length-ratio bins"):
        plot_results.load_length_bins(empty)


def test_plot_writes_a_png(repo_root, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "fig.png"
    args = ["--sweeps", str(repo_root / "results" / "gate_sweeps.csv")]
    args += ["--baselines", str(repo_root / "results" / "heuristic_baselines.md")]
    assert plot_results.main([*args, "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
