# Fallback features on their own (competition train set)

Rows: 48,439. Distinct prompts: 44,418. Share of rows where B won: 0.5054.
Rows that share a prompt are not independent, so the intervals are somewhat too narrow.

## 1. Each rule alone

| Rule | Rows decided | Accuracy when decided | 95 % CI |
|---|---|---|---|
| more digit runs (the fallback used) | 35,500 / 48,439 (73.3%) | 0.5830 | [0.5779, 0.5882] |
| more characters | 48,381 / 48,439 (99.9%) | 0.5837 | [0.5793, 0.5880] |
| more words | 47,850 / 48,439 (98.8%) | 0.5789 | [0.5745, 0.5833] |
| longer average word | 48,389 / 48,439 (99.9%) | 0.5122 | [0.5078, 0.5167] |

## 2. Digit runs versus length

Rows where the two rules pick different responses: 11,298. The digit-run rule is right on 0.4844 of them (95 % CI [0.4752, 0.4936]).

Within bins of |log(chars A / chars B)|:

| Length ratio bin | Digit-run rule: decided, accuracy | Length rule: decided, accuracy |
|---|---|---|
| [0, 0.1) | 4,933, 0.5477 [0.5338, 0.5616] | 7,027, 0.5258 [0.5141, 0.5375] |
| [0.1, 0.3) | 8,929, 0.5483 [0.5380, 0.5586] | 12,103, 0.5409 [0.5321, 0.5498] |
| [0.3, 0.7) | 11,570, 0.5799 [0.5708, 0.5888] | 15,289, 0.6040 [0.5963, 0.6118] |
| >= 0.7 | 10,068, 0.6348 [0.6253, 0.6441] | 13,962, 0.6275 [0.6194, 0.6355] |

## 3. Digit-run rule by language (top 6 by rows)

| Language | Rows | Digit-run rule: decided, accuracy [95 % CI] |
|---|---|---|
| English | 25,211 | 18,904, 0.5815 [0.5744, 0.5885] |
| Russian | 6,455 | 4,512, 0.5836 [0.5691, 0.5979] |
| Chinese | 4,310 | 3,159, 0.5837 [0.5664, 0.6008] |
| Vietnamese | 3,103 | 2,464, 0.5779 [0.5583, 0.5973] |
| German | 1,402 | 951, 0.5741 [0.5425, 0.6052] |
| Japanese | 1,159 | 778, 0.5720 [0.5369, 0.6063] |
