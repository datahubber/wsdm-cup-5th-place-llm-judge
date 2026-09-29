"""`infer.score()` and `scripts/predict.py` against a stub `vllm` module (CPU, no model, no GPU).

The stub returns, for the i-th prompt, the i-th row of tests/fixtures/logprobs.csv as vLLM's
top-k map (token id -> object with `.logprob`). This checks the engine and sampling arguments, the
log-prob read-out and the file outputs. It does not check real vLLM behaviour.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pandas as pd
import predict
import pytest

from wsdm_judge.infer import EngineConfig, score

A_ID, B_ID = ord("A"), ord("B")  # label ids under the character-level test tokenizer


def _top_k(lp_a: float, lp_b: float) -> dict[int, Any]:
    top: dict[int, Any] = {ord("x"): SimpleNamespace(logprob=-5.0)}  # an unrelated token
    if not math.isnan(lp_a):
        top[A_ID] = SimpleNamespace(logprob=lp_a)
    if not math.isnan(lp_b):
        top[B_ID] = SimpleNamespace(logprob=lp_b)
    return top


def install_stub_vllm(monkeypatch: pytest.MonkeyPatch, rows: pd.DataFrame) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine"] = kwargs

        def generate(self, prompts: Sequence[str], params: Any, use_tqdm: bool = True) -> list:
            calls["prompts"], calls["sampling"] = list(prompts), params.kwargs
            tops = [_top_k(a, b) for a, b in zip(rows["lp_a"], rows["lp_b"], strict=True)]
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[t])]) for t in tops]

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    stub = types.ModuleType("vllm")
    stub.LLM, stub.SamplingParams = LLM, SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", stub)
    return calls


def test_score_reads_label_logprobs(monkeypatch, tok, logprobs_path):
    rows = pd.read_csv(logprobs_path)
    calls = install_stub_vllm(monkeypatch, rows)
    got = score([f"p{i}" for i in range(len(rows))], "judge-dir", tok, EngineConfig())
    assert got[0] == (-0.1, -2.4)
    assert got[6] == (-0.3, None) and got[7] == (None, None)
    engine, sampling = calls["engine"], calls["sampling"]
    assert engine["model"] == "judge-dir" and engine["tensor_parallel_size"] == 2
    assert engine["trust_remote_code"] is False
    assert (sampling["max_tokens"], sampling["logprobs"], sampling["temperature"]) == (1, 20, 0.0)


def test_predict_matches_the_cpu_regate(monkeypatch, chat_tok, pairs_path, logprobs_path, tmp_path):
    calls = install_stub_vllm(monkeypatch, pd.read_csv(logprobs_path))
    monkeypatch.setattr(predict, "load_tokenizer", lambda model_path: chat_tok)
    sub, lps = tmp_path / "submission.csv", tmp_path / "logprobs.csv"
    args = ["--data", str(pairs_path), "--model", "judge-dir", "--tau", "0.01"]
    assert predict.main([*args, "--out", str(sub), "--logprobs-out", str(lps)]) == 0

    assert len(calls["prompts"]) == 8
    assert all(p.endswith("<|im_start|>assistant\nAnswer:\n") for p in calls["prompts"])
    assert calls["engine"]["max_model_len"] == 12000
    saved = pd.read_csv(lps)
    assert saved["lp_a"].isna().tolist() == [False] * 7 + [True]
    assert saved["lp_b"].isna().tolist() == [False] * 6 + [True, True]
    # Same decisions as scripts/apply_gate.py on the same log-probs (see test_scripts.py).
    winners = pd.read_csv(sub)["winner"].str[-1].str.upper().tolist()
    assert winners == ["A", "B", "B", "A", "B", "B", "A", "A"]
