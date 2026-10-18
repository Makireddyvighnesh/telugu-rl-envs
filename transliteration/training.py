"""TRL adapter: GRPO trains through the server. Only task IDs go through the dataset.

    trainer = GRPOTrainer(
        model=...,
        train_dataset=Dataset.from_list(task_rows(url, "train", limit=16000, seed=0)),
        environment_factory=lambda: TrainingEnvironment(url, snapshot_id),
        reward_funcs=[env_reward, env_cer, env_exact_match],
        ...
    )

TRL calls ``reset(**row)`` on one environment per rollout before generation; the
returned prompt becomes the user message. ``env_reward`` then sends each completion
to that rollout's environment with ``step`` and uses the server's reward. The trainer
never sees a reference.
"""

import math
import random

from .client import connect
from .models import TransliterationAction


class TrainingEnvironment:
    # TRL exposes public methods as tools; this single-turn task needs only reset.
    def __init__(self, url, snapshot_id):
        self.client = connect(url)
        self.snapshot_id = snapshot_id
        self.task_id = None
        self.result = None

    def reset(self, task_id, **kwargs):
        observation = self.client.reset(task_id=task_id).observation
        if observation.task_id != task_id or observation.snapshot_id != self.snapshot_id:
            raise RuntimeError("Rollout task or snapshot changed")
        self.task_id, self.result = task_id, None
        return observation.prompt

    def _grade(self, answer):
        result = self.client.step(TransliterationAction(answer=answer))
        if not result.done or result.reward is None or not math.isfinite(float(result.reward)):
            raise RuntimeError("Invalid terminal reward")
        self.result = result
        return result

    def _close(self):
        self.client.close()


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"] if isinstance(completion, list) else completion
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content if b.get("type") == "text")


def env_reward(completions, environments, task_id, **kwargs):
    """The environment's reward for each completion, via one step per rollout."""
    scores = []
    for completion, environment, expected in zip(completions, environments, task_id, strict=True):
        if environment.task_id != expected:
            raise RuntimeError("Reward was routed to the wrong rollout task")
        scores.append(float(environment._grade(completion_text(completion)).reward))
    return scores


# Logged with weight 0: they read the result env_reward already got, so they add no calls.
def env_cer(completions, environments, **kwargs):
    return [1.0 - float(e.result.observation.metrics["cer"]) for e in environments]


def env_exact_match(completions, environments, **kwargs):
    return [float(e.result.observation.metrics["exact_match"]) for e in environments]


def task_rows(url, split, limit=None, seed=None):
    """Task IDs of a split, in a seeded shuffle; seed None keeps the snapshot order."""
    with connect(url) as client:
        count = client.num_tasks(split)
        order = list(range(count))
        if seed is not None:
            random.Random(seed).shuffle(order)
        wanted = order[:limit] if limit else order
        rows = {}
        for start in range(0, count, 2048):
            for task in client.get_task_range(split, start, min(start + 2048, count)):
                rows[task["index"]] = task
    return [{"task_id": rows[i]["task_id"]} for i in wanted]
