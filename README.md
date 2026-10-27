# telugu-rl-envs

Reinforcement-learning environments for Telugu, built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv).
Each environment serves one task per episode: `reset()` returns the input, `step()` grades one
answer against a reference the trainer never sees.

| Environment | Input → answer | Reward | Data |
|---|---|---|---|
| [**Transliteration**](transliteration/) (standalone OpenEnv server) | romanized word → Telugu script | `0.8 × (1 − CER) + 0.2 × exact` | Aksharantar, human-romanized sources |
| OCR (`telugu_envs.ocr`) | image crop → text | `1 − CER` | Nayana corpus, Nayana Natural, Sarvam Indic OCR Bench |
| ASR (`telugu_envs.asr`) | audio clip → text | `1 − CER` | FLEURS `te_in` |
| Translation (`telugu_envs.text_env`) | English ↔ Telugu | chrF++ | Samanantar, FLORES+ |

## Results

Qwen2.5-1.5B-Instruct + LoRA, 1000 GRPO steps trained through the transliteration environment
on Kaggle 2×T4. Greedy decoding on all 10,260 Aksharantar Telugu test words:

| | base | GRPO |
|---|---:|---:|
| reward | 0.372 | **0.695** |
| character error rate | 0.547 | **0.204** |
| exact match | 4.6% | **29.1%** |

Adapter, training curves and logs:
[makireddyvighnesh/Qwen2.5-1.5B-Instruct-telugu-translit-grpo](https://huggingface.co/makireddyvighnesh/Qwen2.5-1.5B-Instruct-telugu-translit-grpo).
Base-model numbers and the sampling check are in [`results/baselines.md`](results/baselines.md).

On-policy distillation of that GRPO model into Qwen2.5-0.5B-Instruct + LoRA
([`scripts/train_opd.py`](scripts/train_opd.py), 1000 steps, reverse KL; the student trains on one T4
while the teacher scores its rollouts on the other). Same test:

| | 0.5B base | 0.5B OPD | 1.5B GRPO (teacher) |
|---|---:|---:|---:|
| reward | 0.156 | **0.666** | 0.695 |
| character error rate | 0.806 | **0.229** | 0.204 |
| exact match | 0.5% | **24.6%** | 29.1% |

## Quick start

Run the transliteration environment, with its playground at `http://localhost:8000/web`:

```bash
docker run -p 8000:8000 ghcr.io/makireddyvighnesh/telugu-transliteration-env:latest
```

Or from source:

```bash
pip install -e "./transliteration[train]"
python -m telugu_transliteration.data.prepare --output snapshot
TRANSLIT_SNAPSHOT=snapshot python -m telugu_transliteration.server.app
```

Use it from Python:

```python
from telugu_transliteration.client import connect
from telugu_transliteration.models import TransliterationAction

with connect("http://localhost:8000") as env:
    obs = env.reset(split="test", index=0).observation   # obs.source_text, obs.prompt
    result = env.step(TransliterationAction(answer="నక్షత్రం"))
    print(result.reward, result.observation.metrics)      # cer, wer, exact_match, ...
```

See [`transliteration/README.md`](transliteration/README.md) for the task API, splits and grading.

## Scoring

Both sides are normalized before scoring, following the Sarvam Indic OCR Bench rules:
NFKC/NFC, quote and dash folding, danda/pipe, zero-width joiners, HTML and bullets, and
Telugu digits mapped to ASCII. ASR mode also drops punctuation. `<think>` blocks and an
enclosing code fence are stripped from the answer. Empty answers and answers that loop or
balloon score 0, so a policy can't farm partial credit by repeating itself. CER is capped
at 1 per answer.

## Training

GRPO through the environment server (TRL `environment_factory`; the dataset holds task IDs only):

```bash
TRANSLIT_SNAPSHOT=snapshot python -m telugu_transliteration.server.app &
accelerate launch --multi_gpu --num_processes 2 -m telugu_transliteration.train_grpo \
    --env-url http://127.0.0.1:8000 --out outputs/grpo-env
```

Or in-process, calling the same reward function directly:

```bash
accelerate launch --multi_gpu --num_processes 2 scripts/train_grpo_text.py --out outputs/grpo-text
```

Evaluate an adapter on the test set (greedy, plus 8 samples per word on a 1,000-word subset):

```bash
python scripts/eval_text.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/grpo-env/final \
    --replicas 2 --out outputs/eval
```

On Kaggle, run `pip uninstall -y torchao` after installing: Kaggle ships torchao 0.10, which
current PEFT rejects.

## Data sources

The in-process environments stream from these sources (`telugu_envs.data.SOURCES`):

| Key | Task | Split | Dataset |
|---|---|---|---|
| `nayana_regions` | OCR | train | `Cognitive-Lab/NayanaOCR_Corpus_2025` [te], layout-region crops |
| `nayana_natural` | OCR | eval | `Cognitive-Lab/NayanaOCRBench_Natural` [te], 461 real pages |
| `sarvam` | OCR | test | `sarvamai/indic-ocr-bench`, 300 Telugu blocks |
| `fleurs` | ASR | train / validation / test | `google/fleurs` [te_in] |
| `aksharantar` | transliteration | train / valid / test | `ai4bharat/Aksharantar`, 50.9k / 3.7k / 10.3k human-romanized pairs |
| `samanantar` | translation | train | `ai4bharat/samanantar` [te] |
| `flores` | translation | dev / devtest | `openlanguagedata/flores_plus` (gated) |

## Layout

```
transliteration/      standalone OpenEnv environment (server, playground, client, TRL adapter, Dockerfile)
telugu_envs/
  text.py             normalization, output cleanup, loop detection
  reward.py           CER / WER / chrF rewards
  data.py             streaming sample sources
  base.py             shared single-turn environment, SharedSampler
  ocr/ asr/           OCR and ASR environments
  text_env.py         transliteration and translation environment
  server.py           OpenEnv FastAPI apps for the above
scripts/              GRPO training, evaluation, server smoke test
results/              baseline numbers
tests/
```

## License

The transliteration environment is Apache-2.0. Data stays under its own license: Aksharantar
and FLEURS are CC BY 4.0, Nayana is CC BY-NC 4.0, Samanantar is CC BY-NC.
