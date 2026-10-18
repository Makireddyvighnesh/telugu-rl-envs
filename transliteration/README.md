---
title: Telugu Transliteration
emoji: ✍️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8000
license: apache-2.0
short_description: OpenEnv romanized-to-Telugu transliteration from Aksharantar
tags:
  - openenv
  - reinforcement-learning
  - telugu
  - transliteration
  - indic
datasets:
  - ai4bharat/Aksharantar
---

# Telugu transliteration

Read a Telugu word written in Latin letters, write it in Telugu script, and see how it
scores. This Space serves the human-romanized part of
[Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar), 65,159 Telugu tasks,
as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment you can play in the
browser or train a model against.

Open the playground at `/web`, press Shuffle, and type the word. The reference is revealed
after you score.

## Tasks

| Split | Tasks | Origins |
|---|---:|---|
| `train` | 50,927 | Dakshina |
| `validation` | 3,716 | Dakshina |
| `test` | 10,260 | Dakshina 5,724 · AK-Freq 2,241 · AK-NEI 1,347 · AK-NEF 948 |
| `validation-256` | 256 | frozen eval set: the first 256 validation tasks after a seed-0 shuffle |

Every task is one episode: `reset` serves a romanized word, `step` grades one answer and
ends the episode. Task IDs are stable (`te-translit/<split>/<aksharantar id>`), and the
snapshot is pinned to an Aksharantar revision, so a run can be replayed task for task.

Aksharantar's Telugu train split is otherwise 95% IndicCorp: web-mined, model-romanized,
and about 1 in 8 native words misspelled in a 40-row audit. Those 2.38M pairs are left out.
AK-NEI and AK-NEF are Indian and foreign named entities, the hardest group.

## Reward

`0.8 × (1 − CER) + 0.2 × exact match`, and 0 for an empty or looping answer.

CER is the edit distance over Unicode code points divided by the reference length, capped
at 1. A word rate would be too coarse here: one wrong vowel sign makes a long Telugu word
wrong, so fixing three letters would score the same as leaving it wrong. Before scoring,
both sides are normalized (NFKC/NFC, zero-width joiners and quotes removed, dashes and
dandas folded, Telugu digits mapped to ASCII), and `<think>` blocks and an enclosing code
fence are stripped from the answer. Prose around the answer is not stripped: the reward
should teach a policy to output only the word.

Every result reports `cer`, `wer`, `exact_match`, `empty` and `degenerate` in `metrics`.
The grading policy is `te-translit-cer-v1`.

## Training against it

```python
from telugu_transliteration.training import TrainingEnvironment, env_reward, task_rows

train = Dataset.from_list(task_rows(url, "train", limit=16000, seed=0))   # task IDs only
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=train,
    environment_factory=lambda: TrainingEnvironment(url, snapshot_id),
    reward_funcs=[env_reward],
    ...
)
```

TRL calls `reset(task_id)` on one environment per rollout, which returns the prompt, and
`env_reward` sends each completion back with `step`. The trainer never sees a reference.
`python -m telugu_transliteration.train_grpo --env-url <url>` runs the full recipe.

## API

- `GET /healthz`: the snapshot id
- `GET /manifest`: the snapshot, split counts, and the grading policy
- OpenEnv task discovery (`/telugu_transliteration/num_tasks`, `/task_range`), and
  `reset` / `step` over HTTP and WebSocket

Task discovery and observations never return a reference.

## Run it locally

```bash
python -m telugu_transliteration.data.prepare --output snapshot
TRANSLIT_SNAPSHOT=snapshot python -m telugu_transliteration.server.app
python -m telugu_transliteration.smoke --snapshot snapshot
```

The words are Aksharantar's, **CC BY 4.0**, with credit to AI4Bharat. The environment code
is Apache-2.0. Its structure follows the FineEnvs
[multilingual OCR and ASR environments](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs).
