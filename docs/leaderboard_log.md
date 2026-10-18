# Leaderboard log: my post-processing submissions

This file lists all of my leaderboard submissions that applied a post-processing rule, scored or not, taken from [`results/my_lb_submissions.csv`](../results/my_lb_submissions.csv). It also gives the no-gate references they are compared against. All scores are public-LB accuracy on the 9,997-row public split, so one row is 0.0001. "Rows" means round(Δ × 9997) against the reference named in each section. Rule descriptions in quotes are my own submission descriptions. Teammates' reference scores come from the team's submission list (Kaggle API, pulled 2026-09-29).

Per-row flips were not logged, so comparisons are unpaired; with d flipped rows, a net gain of g rows passes a two-sided sign test at p < 0.05 when g > 1.96·√d.

## Gemma-2 baseline (2024-12-23 to 2024-12-28)

- **Model.** A public Gemma-2-9B 4-bit notebook with public LMSYS LoRA adapters and swap-TTA (c0ldzy's "Improved Weights", which builds on takamichitoda's work). Without post-processing it scored **0.661298** (my submission of 2024-12-23). That is the reference for every row in this section.
- **Gate.** It acts on P(A) − P(B) from the 3-class softmax (A, B, tie), averaged over both response orders. This is a probability difference, not a log-prob margin, so these τ values are not comparable with the Qwen sweeps below.
- **Saved code.** Only the notebook's last version is saved: τ = 0.045, "count number", which counts maximal digit runs. The Dec 23–25 runs are logged only as "count". That they also counted digit runs is an assumption.

### Sweep A: threshold

| τ | 0.02 | 0.03 | 0.04 | 0.045 | 0.05 | 0.06 | 0.07 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|---|---|---|
| "count" (Dec 23–25) | 0.661298 (0) | 0.662998 (+17) | 0.664099 (+28) | | 0.665399 (+41) | 0.664099 (+28) | 0.663499 (+22) | 0.660298 (−10) | 0.651195 (−101) |
| "count number" (Dec 27–28) | | | | 0.665399 (+41) | **0.665899 (+46)** | | | | |

### Fallback feature at τ = 0.05 (all submitted on 2024-12-27)

The last column is each rule's standalone accuracy on the train set, from [`results/heuristic_baselines.md`](../results/heuristic_baselines.md).

| Fallback when the judge is unsure | Public LB | Rows | Train accuracy when the rule decides |
|---|---|---|---|
| more digit runs | 0.665899 | +46 | 0.5830 |
| more characters | 0.664399 | +31 | 0.5837 |
| more words | 0.662998 | +17 | 0.5789 |
| longer average word | 0.660398 | −9 | 0.5122 |

The 15-row gap between digit runs and characters is inside the noise described above, and on the train set the two rules are equally accurate. Only average word length stands out as worse.

### Other variants

| Rule (my description) | Public LB | Rows |
|---|---|---|
| "count > 2 * other, else llm" (the count overrides the judge, no gate; "count" assumed to be digit runs) | 0.635290 | −260 |
| "count > 3 * other, else llm" | 0.641392 | −199 |
| "count > 10 * other, else llm" | 0.655696 | −56 |
| "diff < 0.07 and count > 10" | 0.660998 | −3 |
| "diff < 0.05 and count > 10" | 0.661098 | −2 |
| "diff > 0.07 and count > 10" | 0.661098 | −2 |
| "diff < 0.1 and count > 3*" | 0.660598 | −7 |
| "diff < 0.08 and count > 1.5 *" | 0.663298 | +20 |
| "v9 + only count 4 notations" | 0.663999 | +27 |
| "v9 + updated notation checklist" | 0.664499 | +32 |
| "diff > 0.05, else count positive" (positive words) | 0.663198 | +19 |

Together with Sweep A (10 runs) and the three other fallback features, that makes **24 scored post-processing variants** on the Gemma baseline. Four more runs never produced a score: three decision-tree fallbacks on 5 features (Dec 28), and one rule that combined the digit count with a checklist count (Dec 27).

## Fine-tuned Qwen2.5-14B judge

My first runs of the team's early LoRA checkpoint (`qwen2-5-14b-inferab`), 2024-12-29 to 2024-12-31, scored 0.689706–0.691707. Their descriptions mention only token limits or the label function, not a gate.

### Sweep B: swap + extra-data checkpoint (2025-01-27 and 2025-02-02)

- **Setup.** The `wsdm+lmsys+add_swap` checkpoint, run through my notebook. The saved version uses 8,192-token prompts; the earlier versions are assumed to match. The gate acts on the log-prob margin.
- **Reference.** 0.703811, from the teammate's own notebook with the same checkpoint name. Truncation details may differ, so treat this reference as approximate.

| τ | none | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| Band around P(A) = 0.5 | 0 | ±0.25 pt | ±0.5 pt | ±1.25 pt | ±2.5 pt | ±5.0 pt |
| Public LB | 0.703811 | **0.703911** | 0.703611 | 0.703411 | 0.703010 | 0.702010 |
| Rows | 0 | **+1** | −2 | −4 | −8 | −18 |

Two more runs of this notebook have no score: its version 1 (2025-01-26) and a run described only as "test" (2025-02-02).

### Sweep C: final judge (2025-02-03)

| Run | Public LB | Rows vs. 4,096-token argmax |
|---|---|---|
| Teammate's notebook, default truncation, plain argmax (final 1) | 0.706812 | +9 |
| Teammate's notebook, 4,096-token prompts, plain argmax | 0.705911 | 0 |
| My notebook (a teammate's copy), 4,096 tokens, τ = 0.005 (final 2) | 0.706011 | +1 |
| My notebook (a teammate's copy), 6,400 tokens, τ = 0.002 | 0.706211 | +3 (prompt length differs too) |

The two finals differ in prompt length as well as the gate. Their 1–2-row private gap (0.712627 vs 0.712582) therefore says nothing about the gate.

### Excluded: the 2025-01-13 sweep

This sweep ran on one of the team's chain-of-thought multi-task checkpoints. It scored 0.694808, 0.692707 and 0.695908 at τ = 0.05, 0.08 and 0.02. I cannot tell which checkpoint it used, and the teammate's no-gate runs of that family up to that date span 0.694008–0.697309. The saved (latest) version of that notebook also gates on a two-way softmax probability difference, with a vote over several counts, not on the log-prob margin. The sweep is left out of all conclusions. Three more runs of that notebook have no score: "test" and "softmax test" (2025-01-12) and "post processing: vote on 3 count" (2025-01-19).

### Excluded: other Qwen gate runs (2024-12-31 and 2025-01-03)

| Date | Notebook | My description | Public LB |
|---|---|---|---|
| 2024-12-31 | `vllm-qwen2-5-14b-c14508`, version 1 | "post processing: diff > 0.05 else count numbers" | no score |
| 2025-01-03 | `vllm-qwen2-5-14b-362ffe` | "post processing: diff < 0.05 then count number" (2 runs) | no score |
| 2025-01-03 | `vllm-qwen2-5-14b-362ffe` | "diff < 0.05 then count number" | no score |
| 2025-01-03 | `vllm-qwen2-5-14b-362ffe` | "test" | no score |

