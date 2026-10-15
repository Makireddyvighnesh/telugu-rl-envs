"""Single-turn transcription environment shared by the OCR and ASR tasks.

Episode: ``reset()`` serves one sample (image or audio + instruction);
``step(TranscriptionAction)`` scores the transcription and ends the episode.

GRPO needs G rollouts on the same input. Pass the ``sample_id`` returned by
the first ``reset()`` to the other G-1 resets; recent samples are kept in a
process-wide LRU cache so this works across concurrent sessions.
"""

from __future__ import annotations

import base64
import io
import threading
from collections import OrderedDict
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from telugu_envs.data import Sample, SampleSource
from telugu_envs.reward import transcription_reward


class TranscriptionAction(Action):
    transcription: str = Field(..., description="The model's transcription of the image/audio")


class TranscriptionObservation(Observation):
    sample_id: str = Field(default="", description="Pass to reset(sample_id=...) to replay this sample")
    instruction: str = Field(default="")
    image_png_b64: Optional[str] = Field(default=None, description="Base64 PNG (OCR)")
    audio_wav_b64: Optional[str] = Field(default=None, description="Base64 16-bit PCM WAV (ASR)")
    sampling_rate: Optional[int] = Field(default=None)
    # Populated after step():
    reference: Optional[str] = Field(default=None)
    cer: Optional[float] = Field(default=None)
    wer: Optional[float] = Field(default=None)
    # OpenEnv's serializer drops Observation.metadata, so diagnostics get real fields.
    flags: list[str] = Field(default_factory=list, description="e.g. empty, tail_loop, length_explosion")
    cleaned_prediction: Optional[str] = Field(default=None, description="Transcription after wrapper stripping")
    source: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


class TranscriptionState(State):
    sample_id: Optional[str] = None
    source: Optional[str] = None
    reward: Optional[float] = None


def encode_image(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def encode_audio(audio: np.ndarray, sr: int) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


class SharedSampler:
    """Thread-safe sample stream shared by every env instance in the process."""

    def __init__(self, source: SampleSource, cache_size: int = 4096):
        self._source = source
        self._iters: dict[Optional[int], Any] = {}
        self._cache: OrderedDict[str, Sample] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def next(self, seed: Optional[int]) -> Sample:
        with self._lock:
            it = self._iters.get(seed)
            if it is None:
                it = self._iters[seed] = self._source(seed)
            try:
                sample = next(it)
            except StopIteration:  # new epoch
                it = self._iters[seed] = self._source(seed)
                sample = next(it)
            self._remember(sample)
            return sample

    def get(self, sample_id: str) -> Sample:
        with self._lock:
            if sample_id not in self._cache:
                raise KeyError(f"sample {sample_id!r} not in cache (evicted or never served)")
            self._cache.move_to_end(sample_id)
            return self._cache[sample_id]

    def _remember(self, sample: Sample) -> None:
        self._cache[sample.id] = sample
        self._cache.move_to_end(sample.id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


class TranscriptionEnvironment(Environment):
    """Base class; subclasses set ``MODE``, ``INSTRUCTION`` and ``_encode``."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    MODE = "ocr"
    INSTRUCTION = ""

    def __init__(self, sampler: SharedSampler, cer_weight: float = 1.0, instruction: Optional[str] = None):
        super().__init__()
        self._sampler = sampler
        self._cer_weight = cer_weight
        self._instruction = instruction or self.INSTRUCTION
        self._sample: Optional[Sample] = None
        self._state = TranscriptionState(episode_id=str(uuid4()), step_count=0)

    def _encode(self, sample: Sample) -> dict:
        raise NotImplementedError

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionObservation:
        self._sample = self._sampler.get(sample_id) if sample_id else self._sampler.next(seed)
        self._state = TranscriptionState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            sample_id=self._sample.id,
            source=self._sample.meta.get("source"),
        )
        return TranscriptionObservation(
            sample_id=self._sample.id,
            instruction=self._instruction,
            done=False,
            reward=None,
            **self._encode(self._sample),
        )

    def step(self, action: TranscriptionAction, timeout_s: Optional[float] = None, **kwargs: Any) -> TranscriptionObservation:
        if self._sample is None or self._state.reward is not None:
            return TranscriptionObservation(done=True, reward=0.0, error="call reset() before step()")
        self._state.step_count += 1
        result = transcription_reward(self._sample.reference, action.transcription, self.MODE, self._cer_weight)
        self._state.reward = result.reward
        return TranscriptionObservation(
            sample_id=self._sample.id,
            instruction=self._instruction,
            done=True,
            reward=result.reward,
            reference=self._sample.reference,
            cer=result.cer,
            wer=result.wer,
            flags=result.flags,
            cleaned_prediction=result.prediction,
            source=self._sample.meta.get("source"),
            metadata=dict(self._sample.meta),
        )

    @property
    def state(self) -> TranscriptionState:
        return self._state
