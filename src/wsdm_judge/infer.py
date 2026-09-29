"""Score pairs with vLLM: generate one token and read the log-probabilities of "A" and "B".

The judge answers with a single token, so one decoding step per pair is enough. The step returns
the top-20 log-probabilities, and the two label tokens are read from them. The model's sampled
text is ignored.

`vllm` and `transformers` are imported inside the functions that need them, so the rest of the
package (and the test suite) runs on a CPU-only machine without either installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wsdm_judge.prompt import Tokenizer

LABELS = ("A", "B")

# (logprob of "A", logprob of "B"). None when the token is not in the returned top-k.
LabelLogprobs = tuple[float | None, float | None]


@dataclass(frozen=True)
class EngineConfig:
    """vLLM settings of the Kaggle run (2x T4, 16 GB each, no bf16 support)."""

    tensor_parallel_size: int = 2
    dtype: str = "half"
    gpu_memory_utilization: float = 0.99
    # 14.7B parameters in fp16 take about 29 GB, close to all the memory of two T4s. Offloading
    # 4 GB of weights to CPU RAM leaves room for the KV cache.
    cpu_offload_gb: float = 4
    swap_space: float = 1
    max_model_len: int = 12000
    max_num_seqs: int = 16
    enforce_eager: bool = True
    # Every prompt starts with the same system rubric, so its KV blocks are computed once.
    enable_prefix_caching: bool = True
    top_logprobs: int = 20
    # The Kaggle notebook passed True. Qwen2.5 is supported natively by vLLM and `transformers`,
    # so it is not needed, and True would run any Python code shipped with the weights.
    trust_remote_code: bool = False


DEFAULT_ENGINE = EngineConfig()


class TransformersTokenizer:
    """Adapts a `transformers` tokenizer to `prompt.Tokenizer` and `prompt.RenderChat`."""

    def __init__(self, tokenizer: Any) -> None:
        self.inner = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.inner(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.inner.decode(list(ids), skip_special_tokens=False))

    def render_chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return str(
            self.inner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )


def load_tokenizer(model_path: str) -> TransformersTokenizer:
    from transformers import AutoTokenizer

    return TransformersTokenizer(AutoTokenizer.from_pretrained(model_path))


def label_token_ids(tok: Tokenizer) -> tuple[int, int]:
    """Token ids of "A" and "B" as the judge emits them (last id if a label splits)."""
    return tok.encode(LABELS[0])[-1], tok.encode(LABELS[1])[-1]


def ab_logprobs(top: Mapping[int, Any] | None, a_id: int, b_id: int) -> LabelLogprobs:
    """Read the two label log-probabilities from one step's top-k map.

    vLLM maps token id -> `Logprob` object (with a `.logprob` field). Plain floats are accepted too,
    which keeps the tests free of vLLM.
    """
    if not top:
        return None, None

    def value(token_id: int) -> float | None:
        if token_id not in top:
            return None
        entry = top[token_id]
        return float(getattr(entry, "logprob", entry))

    return value(a_id), value(b_id)


def score(
    prompts: Sequence[str], model_path: str, tok: Tokenizer, cfg: EngineConfig = DEFAULT_ENGINE
) -> list[LabelLogprobs]:
    """Run the judge once over all prompts (needs a GPU and `pip install .[infer]`)."""
    import vllm

    llm = vllm.LLM(
        model=model_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        dtype=cfg.dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        cpu_offload_gb=cfg.cpu_offload_gb,
        swap_space=cfg.swap_space,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        enforce_eager=cfg.enforce_eager,
        enable_prefix_caching=cfg.enable_prefix_caching,
        trust_remote_code=cfg.trust_remote_code,
        disable_log_stats=True,
    )
    params = vllm.SamplingParams(
        n=1,
        temperature=0.0,
        top_k=1,
        max_tokens=1,
        logprobs=cfg.top_logprobs,
        skip_special_tokens=False,
    )
    outputs = llm.generate(list(prompts), params, use_tqdm=True)
    a_id, b_id = label_token_ids(tok)
    result: list[LabelLogprobs] = []
    for out in outputs:
        steps = out.outputs[0].logprobs if out.outputs else None
        result.append(ab_logprobs(steps[0] if steps else None, a_id, b_id))
    return result
