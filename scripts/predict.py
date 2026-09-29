"""Kaggle-style inference: prompts -> vLLM label log-probabilities -> gate -> submission.csv.

Needs a CUDA GPU, `pip install ".[infer]"` and the judge weights (the team's merged Qwen2.5-14B
model, released on Kaggle; public availability is not guaranteed). This rewrite has run only
against a stub `vllm` (tests/test_predict_stub.py), never on a GPU; see README, Limitations. The
log-probabilities are saved too, so `apply_gate.py` can re-run the post-processing on CPU with
another tau.

    python scripts/predict.py --data test.parquet --model /path/to/judge --tau 0.005
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from wsdm_judge.data import read_table
from wsdm_judge.infer import EngineConfig, load_tokenizer, score
from wsdm_judge.postprocess import decide_all, reason_counts, to_label
from wsdm_judge.prompt import build_prompt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="pairs with id, prompt, response_a, response_b")
    ap.add_argument("--model", required=True, help="local path of the merged judge model")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--logprobs-out", default="logprobs.csv")
    ap.add_argument("--tau", type=float, default=0.005, help="logprob-margin gate; 0 = argmax")
    ap.add_argument("--max-length", type=int, default=4096, help="prompt token budget")
    ap.add_argument("--max-model-len", type=int, default=12000)
    args = ap.parse_args(argv)

    df = read_table(args.data)
    tok = load_tokenizer(args.model)
    prompts = [
        build_prompt(q, a, b, tok, tok.render_chat, args.max_length)
        for q, a, b in zip(df["prompt"], df["response_a"], df["response_b"], strict=True)
    ]
    lps = score(prompts, args.model, tok, EngineConfig(max_model_len=args.max_model_len))
    lp_a, lp_b = [x[0] for x in lps], [x[1] for x in lps]
    pd.DataFrame({"id": df["id"], "lp_a": lp_a, "lp_b": lp_b}).to_csv(
        args.logprobs_out, index=False
    )

    decisions = decide_all(
        lp_a, lp_b, df["response_a"].tolist(), df["response_b"].tolist(), args.tau
    )
    sub = pd.DataFrame({"id": df["id"], "winner": [to_label(d.winner) for d in decisions]})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows); decisions by reason: {reason_counts(decisions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
