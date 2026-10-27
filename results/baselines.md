# Baselines: Telugu transliteration (Aksharantar test, human sources)

`scripts/eval_text.py`, Kaggle 2×T4, 2 replicas, transformers fp16.
Greedy decoding on all 10,260 test pairs; GRPO-signal check = 8 samples at
temperature 0.9 on a 1,000-pair subset. Reward = 0.8 × (1 − CER) + 0.2 × exact.

## Greedy

| Model | Reward | CER | Exact | chrF | Flagged |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.156 | 0.806 | 0.50% | 0.115 | 31 |
| Qwen2.5-1.5B-Instruct | 0.372 | 0.547 | 4.63% | 0.269 | 0 |
| Qwen2.5-3B-Instruct | 0.432 | 0.480 | 8.29% | 0.332 | 15 |
| Qwen2.5-7B-Instruct | 0.514 | 0.391 | 13.72% | 0.401 | 21 |

### By origin (reward / exact)

| Model | Dakshina (5,724) | AK-Freq (2,241) | AK-NEI (1,347) | AK-NEF (948) |
|---|---|---|---|---|
| 0.5B | 0.162 / 0.63% | 0.172 / 0.31% | 0.128 / 0.45% | 0.124 / 0.21% |
| 1.5B | 0.386 / 6.03% | 0.413 / 3.75% | 0.334 / 2.52% | 0.243 / 1.27% |
| 3B | 0.444 / 9.66% | 0.501 / 7.90% | 0.376 / 6.16% | 0.282 / 4.01% |
| 7B | 0.512 / 15.1% | 0.565 / 12.7% | 0.480 / 11.7% | 0.457 / 10.5% |

## GRPO signal (k = 8, T = 0.9, 1,000 groups)

| Model | Mean reward | Best of 8 | Exact in any of 8 | Groups with signal | All-zero groups | Within-group std |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 0.058 | 0.230 | 0.8% | 73.8% | 26.2% | 0.084 |
| 1.5B | 0.256 | 0.509 | 8.0% | 95.5% | 4.3% | 0.171 |
| 3B | 0.326 | 0.572 | 14.9% | 93.6% | 5.1% | 0.176 |
| 7B | 0.407 | 0.645 | 22.2% | 94.8% | 3.0% | 0.187 |

Takeaways: 1.5B is the GRPO candidate (signal in 95% of groups, best-of-8 well
above greedy). Sampling at T = 0.9 costs ~0.12 reward vs greedy, so train cooler.

3B and 7B (OPD teacher candidates): fp16; 3B as 2 replicas, 7B as one replica sharded across
both T4s (`--device-map auto`). Neither beats the GRPO-trained 1.5B (reward 0.695, exact 29.1%),
so an untuned 3B/7B teacher would pull a 1.5B student down, not up.

## On-policy distillation: 1.5B GRPO -> 0.5B

`scripts/train_opd.py`, 1000 steps, 64 completions/step, reverse KL at T = 1.0, LoRA r = 16, lr 1e-4,
6,156 s on 2xT4. Same greedy test and sampling check as above:

| Model | Reward | CER | Exact | chrF | Best of 8 | Exact in any of 8 |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B OPD | 0.666 | 0.229 | 24.58% | 0.515 | 0.705 | 29.3% |
| 1.5B GRPO (teacher) | 0.695 | 0.204 | 29.05% | 0.527 | 0.741 | 35.0% |

By origin (reward / exact): Dakshina 0.683 / 26.7%, AK-Freq 0.758 / 33.3%, AK-NEI 0.554 / 12.9%,
AK-NEF 0.506 / 8.0%. Validation env reward: 0.382 at step 50, 0.620 at 250, 0.671 at 650, 0.673 at 1000.
