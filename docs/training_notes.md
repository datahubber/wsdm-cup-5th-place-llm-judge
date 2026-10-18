# The judge: training notes

Teammate Chen Yulin ran the fine-tuning of the Qwen2.5-14B judge and every training run on this page. I added the head+tail truncation, the A/B response swap and the soft labels: the three steps marked "added by me" below. The training code is public in his repo [lcy80366872/kaggle-project/wsdm](https://github.com/lcy80366872/kaggle-project/tree/main/wsdm). It builds on Raja Biswas's MIT-licensed [Eedi 1st-place scaffold](https://github.com/rbiswasfc/eedi-mining-misconceptions). This summary comes from his code, his run configs, the team's submission list (Kaggle API, pulled 2026-09-29) and the team's documentation. The training code is not in this repo; this page is here so the inference side makes sense.

## Leaderboard progression (public LB)

| Step | Public LB | Date |
|---|---|---|
| Qwen2.5-14B-Instruct zero-shot, 3 runs | 0.598179–0.625387 | 2024-12-28 |
| + LoRA judge | 0.690707 | 2024-12-28 |
| + head+tail truncation to 8,192 tokens (added by me) | 0.693808 | 2025-01-01 |
| + A/B response swap (WSDM data; added by me) | 0.696709 | 2025-01-24 |
| + LMSYS and extra preference data (with swap) | 0.703811 | 2025-01-26 |
| + stage 2: smoothed labels and soft pseudo-labels (added by me) | 0.706812 | 2025-02-03 |
| **Private LB**, same model | **0.712627** | rerun 2025-03-10 |

This is not a clean ablation. Each step changed data, training or truncation at once, and every step was chosen on the public LB.

## Recipe

- **Data.** WSDM train, LMSYS Chatbot Arena data and an extra preference set, deduplicated to 96,462 pairs. Every pair appears in both A/B orders (192,924 rows), with both orders in the same batch, to counter position bias.
- **Prompt.** The rubric, tagged sections and head+tail truncation that [`prompt.py`](../src/wsdm_judge/prompt.py) rebuilds for inference.
- **Model.** Qwen2.5-14B-Instruct with LoRA (r = 64, α = 128, dropout 0.01) on all attention and MLP projections, trained in 16-bit precision. The final model used no 4-bit quantization.
- **Head.** There is no classifier head. The loss is a 2-way cross-entropy over the LM logits of "A" and "B" at the last position, so vLLM serves the merged model as a plain causal LM.
- **Optimisation.** 8-bit AdamW, LR 1e-5 for LoRA A and 5e-5 for LoRA B, cosine decay to 10 % after 2 % warmup, one epoch.
- **Stage 2: soft labels.**
  - The stage-1 model's A/B probabilities on 26,468 unlabelled pairs become soft pseudo-labels.
  - The model is retrained on the labelled pairs with smoothed 0.95/0.05 targets, plus the pseudo-labelled pairs.
  - This target construction is not in the teammate's public code release, so the 0.703 → 0.706 step cannot be reproduced from it.

## Compute

- **Inference.** Kaggle, 2× T4 (16 GB each). The team documentation reports about 4.5 h for the test set.
- **Training.** The team documentation gives about 12 h (stage 1) plus 16 h (stage 2) on 6× A100 80 GB. The run logs show 2, 4 or 8 DDP processes.

## Experiments that did not go into the final model

| Idea | Public LB | Reference |
|---|---|---|
| Zero-shot Qwen2.5-14B (3 runs) / Qwen2.5-72B (1 run) | 0.598–0.625 / 0.620 | 0.691, fine-tuned 14B |
| One-shot example at inference only | 0.6876 | 0.6907, same model without it |
| One-shot example in training (2 runs) / in training and inference | 0.6903, 0.6910 / 0.6893 | 0.6938, same setup without it |
| An R1-distilled Qwen-14B reasoning model as the base (4,096 / 8,192-token prompts) | 0.6837 / 0.6858 | Team documentation: CV +0.001, LB −0.003 against a matched run, which the log does not identify |
| Auxiliary chain-of-thought loss on model-generated rationales (best of 13 scored runs) | 0.6976 | Matched non-CoT runs: +12 rows once (0.6976 vs 0.6964, WSDM + LMSYS + extra data) and −21 rows once (0.6930 vs 0.6951, WSDM + swap, 4,096 tokens). Team documentation: about +0.003, not adopted |
| Contrastive auxiliary loss; DPO + BCE; a second epoch | not identified in the log | Team documentation: no LB gain |

## A separate line

Another teammate ran a separate ensemble line that reached 0.685205 on 2024-12-25.
