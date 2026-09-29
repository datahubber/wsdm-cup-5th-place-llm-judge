from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from wsdm_judge.infer import EngineConfig, TransformersTokenizer, ab_logprobs, label_token_ids


def test_import_does_not_pull_in_vllm_or_transformers():
    code = (
        "import sys, wsdm_judge.infer;print('vllm' in sys.modules, 'transformers' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_ab_logprobs_reads_vllm_objects_and_floats():
    top = {65: SimpleNamespace(logprob=-0.2), 66: -1.5, 7: SimpleNamespace(logprob=-3.0)}
    assert ab_logprobs(top, 65, 66) == (-0.2, -1.5)


def test_ab_logprobs_missing_tokens():
    assert ab_logprobs({65: -0.1}, 65, 66) == (-0.1, None)
    assert ab_logprobs({}, 65, 66) == (None, None)
    assert ab_logprobs(None, 65, 66) == (None, None)


def test_label_token_ids(tok):
    assert label_token_ids(tok) == (ord("A"), ord("B"))


class FakeTokenizer:
    """The three methods of a `transformers` tokenizer that TransformersTokenizer calls."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return body + ("[assistant]" if add_generation_prompt else "")


def test_transformers_adapter():
    tok = TransformersTokenizer(FakeTokenizer())
    assert tok.decode(tok.encode("héllo")) == "héllo"
    assert tok.render_chat("S", "U") == "[system]S[user]U[assistant]"


def test_engine_defaults_match_the_kaggle_run():
    cfg = EngineConfig()
    got = (cfg.tensor_parallel_size, cfg.dtype, cfg.cpu_offload_gb, cfg.top_logprobs)
    assert got == (2, "half", 4, 20)
    assert cfg.trust_remote_code is False
