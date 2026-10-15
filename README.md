# telugu-rl-envs

OpenEnv-compatible reinforcement-learning environments for Telugu:

- **OCR** — `reset()` returns a document/scene image plus an instruction; `step()` scores the model's transcription by character error rate (CER).
- **ASR** — `reset()` returns an audio clip plus an instruction; `step()` scores the transcription by CER/WER.

Text is normalized before scoring (Unicode NFC, punctuation, numerals).

## Data sources (streamed)

| Source key | Task | Role | Dataset | Unit |
|---|---|---|---|---|
| `nayana_regions` | OCR | train | `Cognitive-Lab/NayanaOCR_Corpus_2025` [te], 45.7k synthetic pages (~40 GB) | single layout-region crop; GT = rendered text |
| `nayana_natural` | OCR | eval | `Cognitive-Lab/NayanaOCRBench_Natural` [te], 461 real human-verified pages | text-block crop |
| `sarvam` | OCR | held-out test | `sarvamai/indic-ocr-bench`, 300 Telugu blocks | block crop |
| `fleurs` | ASR | train / validation / test | `google/fleurs` [te_in] parquet | utterance (≤30 s) |

Nayana pages are 3300×2550 and carry per-region bboxes, so the OCR env serves
region crops rather than whole pages: smaller images, exact references, and a
natural difficulty knob (`min_chars`/`max_chars`).

## Reward

`reward = 1 - CER` (optionally blended with WER via `cer_weight`), capped to
[0, 1]. Before scoring, both sides go through `telugu_envs.text.normalize`,
which follows the Sarvam Indic OCR Bench content normalization (NFKC/NFC,
quote/dash folding, danda/pipe, ZWJ/ZWNJ, HTML/bullets) plus Telugu-digit
folding; ASR mode also drops punctuation. Model wrappers (`<think>…</think>`,
code fences) are stripped. Empty or degenerate outputs (tail loops, length
explosion) get 0 so GRPO can't farm partial credit by repeating.

## Usage

```python
from telugu_envs.base import SharedSampler, TranscriptionAction
from telugu_envs.data import nayana_regions
from telugu_envs.ocr.environment import TeluguOCREnvironment

env = TeluguOCREnvironment(SharedSampler(nayana_regions()))
obs = env.reset(seed=0)            # obs.image_png_b64, obs.instruction, obs.sample_id
obs = env.step(TranscriptionAction(transcription="..."))
obs.reward, obs.cer, obs.wer, obs.reference
```

For GRPO groups, pass the first rollout's `sample_id` to the other resets:
`env.reset(sample_id=obs.sample_id)`.

Serve over OpenEnv (WebSocket sessions at `/ws`):

```bash
TELUGU_SOURCE=nayana_regions uvicorn telugu_envs.server:ocr_app --port 8000
TELUGU_SOURCE=fleurs TELUGU_SPLIT=train uvicorn telugu_envs.server:asr_app --port 8001
```

## Layout

```
telugu_envs/
  text.py    # normalization, output cleanup, loop detection
  reward.py  # CER / WER / transcription_reward
  data.py    # streaming sample sources
  base.py    # shared single-turn env, actions/observations, SharedSampler
  ocr/ asr/  # task environments
  server.py  # OpenEnv FastAPI apps
tests/
```

## Roadmap

- [x] Text normalization + CER/WER reward
- [x] OCR environment
- [x] ASR environment
- [ ] Validate streaming against real data (blocked: HF CDN not reachable from the dev container)
- [ ] Baseline evaluation script
- [ ] GRPO training (TRL + LoRA)
- [ ] Results: before/after curves
