"""OpenEnv FastAPI servers for the Telugu OCR and ASR environments.

    TELUGU_SOURCE=nayana_regions uvicorn telugu_envs.server:ocr_app --port 8000
    TELUGU_SOURCE=fleurs TELUGU_SPLIT=train uvicorn telugu_envs.server:asr_app --port 8001
    TELUGU_SOURCE=aksharantar uvicorn telugu_envs.server:text_app --port 8002

Env vars:
    TELUGU_SOURCE   one of telugu_envs.data.SOURCES (default: nayana_regions / fleurs)
    TELUGU_SPLIT    split for sources that take one (sarvam, fleurs)
    TELUGU_MAX_ENVS max concurrent WebSocket sessions (default 64)
"""

from __future__ import annotations

import os
from functools import lru_cache

from openenv.core.env_server.http_server import create_app

from telugu_envs.asr.environment import TeluguASREnvironment
from telugu_envs.base import SharedSampler, TranscriptionAction, TranscriptionObservation
from telugu_envs.data import SOURCES
from telugu_envs.ocr.environment import TeluguOCREnvironment
from telugu_envs.text_env import TeluguTextEnvironment, TextAction, TextObservation


@lru_cache(maxsize=None)
def _sampler(default_source: str) -> SharedSampler:
    name = os.environ.get("TELUGU_SOURCE", default_source)
    kwargs = {"split": os.environ["TELUGU_SPLIT"]} if "TELUGU_SPLIT" in os.environ else {}
    return SharedSampler(SOURCES[name](**kwargs))


def make_ocr_env() -> TeluguOCREnvironment:
    return TeluguOCREnvironment(_sampler("nayana_regions"))


def make_asr_env() -> TeluguASREnvironment:
    return TeluguASREnvironment(_sampler("fleurs"))


def make_text_env() -> TeluguTextEnvironment:
    return TeluguTextEnvironment(_sampler("aksharantar"))


_MAX_ENVS = int(os.environ.get("TELUGU_MAX_ENVS", "64"))


def __getattr__(name: str):
    # Build apps lazily so importing this module for one task doesn't build the other.
    if name == "ocr_app":
        return create_app(make_ocr_env, TranscriptionAction, TranscriptionObservation, env_name="telugu_ocr", max_concurrent_envs=_MAX_ENVS)
    if name == "asr_app":
        return create_app(make_asr_env, TranscriptionAction, TranscriptionObservation, env_name="telugu_asr", max_concurrent_envs=_MAX_ENVS)
    if name == "text_app":
        return create_app(make_text_env, TextAction, TextObservation, env_name="telugu_text", max_concurrent_envs=_MAX_ENVS)
    raise AttributeError(name)
