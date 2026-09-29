"""End-to-end runs of the CPU scripts on the synthetic fixtures."""

from __future__ import annotations

import apply_gate
import heuristic_baselines
import pandas as pd


def test_heuristic_baselines_report(pairs_path, tmp_path, capsys):
    out = tmp_path / "report.md"
    assert heuristic_baselines.main(["--data", str(pairs_path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## 1. Each rule alone" in text
    assert "more digit runs" in text
    assert "Rows: 8." in text


def test_apply_gate_writes_submission_and_sweep(pairs_path, logprobs_path, tmp_path, capsys):
    sub = tmp_path / "submission.csv"
    args = ["--data", str(pairs_path), "--logprobs", str(logprobs_path), "--tau", "0.01"]
    assert apply_gate.main([*args, "--out", str(sub), "--sweep", "0,0.01"]) == 0
    got = pd.read_csv(sub)
    assert list(got.columns) == ["id", "winner"]
    assert got["winner"].str[-1].str.upper().tolist() == ["A", "B", "B", "A", "B", "B", "A", "A"]
    printed = capsys.readouterr().out
    assert "accuracy at tau=0.01: 0.7500 (8 rows)" in printed
    assert "| 0.01 | 66.67% | 3 | 2 | 1 | 0.8333 |" in printed
