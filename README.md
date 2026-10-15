# telugu-rl-envs

OpenEnv-compatible reinforcement-learning environments for Telugu:

- **OCR** — `reset()` returns a document/scene image plus an instruction; `step()` scores the model's transcription by character error rate (CER).
- **ASR** — `reset()` returns an audio clip plus an instruction; `step()` scores the transcription by CER/WER.

Text is normalized before scoring (Unicode NFC, punctuation, numerals).

## Data sources (streamed)

| Task | Split | Source |
|------|-------|--------|
| OCR  | train / eval | Nayana Telugu OCR |
| OCR  | held-out test | Sarvam OCR bench (Telugu, ~300 rows) |
| ASR  | train / eval / test | FLEURS `te_in` |

## Layout

```
telugu_envs/
  ocr/   # OCR environment
  asr/   # ASR environment
scripts/ # baseline eval, GRPO training
tests/
```

## Roadmap

- [ ] Validate streaming access to all data sources
- [ ] Text normalization + CER/WER reward
- [ ] OCR environment
- [ ] ASR environment
- [ ] Baseline evaluation script
- [ ] GRPO training (TRL + LoRA)
- [ ] Results: before/after curves
