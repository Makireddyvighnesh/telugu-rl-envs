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
| `aksharantar` | Transliteration | train / valid / test | `ai4bharat/Aksharantar` `tel.zip`, human-romanized sources only by default: 50.9k / 3.7k / 10.3k pairs (`sources=None` adds the 2.3M web-mined pairs; `words_per_task=(lo,hi)` joins random words into phrases) | romanized → Telugu script |
| `samanantar` | Translation | train | `ai4bharat/samanantar` [te], streamed (CC BY-NC; web-mined, noisy references) | en ↔ te sentence |
| `flores` | Translation | eval (dev / devtest) | `openlanguagedata/flores_plus` (gated: accept terms) | en ↔ te sentence |
| `fleurs` | ASR | train / validation / test | `google/fleurs` [te_in] parquet | utterance (≤30 s); ~10% of refs keep English terms in Latin script (`exclude_latin=True` drops them) |

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

Text tasks (`telugu_envs.text_env.TeluguTextEnvironment`) reward transliteration with
`0.8 × (1 − CER) + 0.2 × exact match` and translation with chrF++.

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

## Notes

In sandboxed networks where Hugging Face's Xet storage (`cas-server.xethub.hf.co`)
returns 401, set `HF_HUB_DISABLE_XET=1` to download through the regular CDN.

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
- [x] Validate streaming against real data (`RUN_HF_TESTS=1 pytest tests/test_integration_hf.py`; end-to-end: `scripts/smoke_test_server.py`)
- [x] Text environment: transliteration + translation
- [x] Validate Aksharantar on real data, audit sources (see `aksharantar` docstring)
- [ ] Validate FLORES (gated: accept terms)
- [ ] Baseline evaluation script
- [ ] GRPO training (TRL + LoRA)
- [ ] Results: before/after curves
