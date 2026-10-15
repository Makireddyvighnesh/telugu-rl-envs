# Baselines: Telugu transliteration (Aksharantar test, human sources)

`scripts/eval_text.py`, Kaggle 2×T4, 2 replicas, transformers fp16.
Greedy decoding on all 10,260 test pairs; GRPO-signal check = 8 samples at
temperature 0.9 on a 1,000-pair subset. Reward = 0.8 × (1 − CER) + 0.2 × exact.

## Greedy

| Model | Reward | CER | Exact | chrF | Flagged |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.156 | 0.806 | 0.50% | 0.115 | 31 |
| Qwen2.5-1.5B-Instruct | 0.372 | 0.547 | 4.63% | 0.269 | 0 |

### By origin (reward / exact)

| Model | Dakshina (5,724) | AK-Freq (2,241) | AK-NEI (1,347) | AK-NEF (948) |
|---|---|---|---|---|
| 0.5B | 0.162 / 0.63% | 0.172 / 0.31% | 0.128 / 0.45% | 0.124 / 0.21% |
| 1.5B | 0.386 / 6.03% | 0.413 / 3.75% | 0.334 / 2.52% | 0.243 / 1.27% |

## GRPO signal (k = 8, T = 0.9, 1,000 groups)

| Model | Mean reward | Best of 8 | Exact in any of 8 | Groups with signal | All-zero groups | Within-group std |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 0.058 | 0.230 | 0.8% | 73.8% | 26.2% | 0.084 |
| 1.5B | 0.256 | 0.509 | 8.0% | 95.5% | 4.3% | 0.171 |

Takeaways: 1.5B is the GRPO candidate (signal in 95% of groups, best-of-8 well
above greedy). Sampling at T = 0.9 costs ~0.12 reward vs greedy, so train cooler.
