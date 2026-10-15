"""Text-to-text environment: Telugu transliteration and translation.

Same single-turn contract as the OCR/ASR environments (reset -> one step ->
done), but the observation is text and the reward depends on the task:

* transliteration (romanized -> Telugu script): 0.8 * (1 - CER) + 0.2 * exact
* translation (English <-> Telugu): chrF++
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

from telugu_envs.base import SharedSampler, TranscriptionState
from telugu_envs.data import Sample
from telugu_envs.reward import text_reward

INSTRUCTIONS = {
    "transliteration": (
        "Transliterate the following romanized Telugu into Telugu script. "
        "Output only the Telugu text.\n\n{text}"
    ),
    ("translation", "eng-tel"): (
        "Translate the following English text into Telugu. Output only the translation.\n\n{text}"
    ),
    ("translation", "tel-eng"): (
        "Translate the following Telugu text into English. Output only the translation.\n\n{text}"
    ),
}


class TextAction(Action):
    answer: str = Field(..., description="The model's transliteration / translation")


class TextObservation(Observation):
    sample_id: str = Field(default="")
    task: str = Field(default="")
    source_text: str = Field(default="", description="The text to transform")
    prompt: str = Field(default="", description="Ready-to-use prompt: instruction + source text")
    # Populated after step():
    reference: Optional[str] = None
    cer: Optional[float] = None
    chrf: Optional[float] = None
    exact: Optional[bool] = None
    flags: list[str] = Field(default_factory=list)
    cleaned_prediction: Optional[str] = None
    error: Optional[str] = None


class TeluguTextEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, sampler: SharedSampler):
        super().__init__()
        self._sampler = sampler
        self._sample: Optional[Sample] = None
        self._state = TranscriptionState(episode_id=str(uuid4()), step_count=0)

    @staticmethod
    def _task(sample: Sample) -> str:
        return sample.meta.get("task", "transliteration")

    @classmethod
    def prompt_for(cls, sample: Sample) -> str:
        task = cls._task(sample)
        key = task if task == "transliteration" else (task, sample.meta.get("direction", "eng-tel"))
        return INSTRUCTIONS[key].format(text=sample.text)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TextObservation:
        self._sample = self._sampler.get(sample_id) if sample_id else self._sampler.next(seed)
        self._state = TranscriptionState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            sample_id=self._sample.id,
            source=self._sample.meta.get("source"),
        )
        return TextObservation(
            sample_id=self._sample.id,
            task=self._task(self._sample),
            source_text=self._sample.text,
            prompt=self.prompt_for(self._sample),
            done=False,
            reward=None,
        )

    def step(self, action: TextAction, timeout_s: Optional[float] = None, **kwargs: Any) -> TextObservation:
        if self._sample is None or self._state.reward is not None:
            return TextObservation(done=True, reward=0.0, error="call reset() before step()")
        self._state.step_count += 1
        task = self._task(self._sample)
        r = text_reward(self._sample.reference, action.answer, task)
        self._state.reward = r.reward
        return TextObservation(
            sample_id=self._sample.id,
            task=task,
            source_text=self._sample.text,
            done=True,
            reward=r.reward,
            reference=self._sample.reference,
            cer=r.cer,
            chrf=r.chrf,
            exact=r.exact,
            flags=r.flags,
            cleaned_prediction=r.prediction,
        )

    @property
    def state(self) -> TranscriptionState:
        return self._state
