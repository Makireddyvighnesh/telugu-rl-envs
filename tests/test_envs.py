import base64
import io

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from telugu_envs.asr.environment import TeluguASREnvironment
from telugu_envs.base import SharedSampler, TranscriptionAction
from telugu_envs.data import Sample, from_list
from telugu_envs.ocr.environment import TeluguOCREnvironment

REFS = ["తెలుగు భాష", "నమస్తే మిత్రమా", "హైదరాబాద్ నగరం"]


def ocr_sampler():
    return SharedSampler(
        from_list([Sample(id=f"s{i}", reference=r, image=Image.new("RGB", (64, 32), "white"), meta={"source": "test"}) for i, r in enumerate(REFS)])
    )


def asr_sampler():
    audio = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)).astype(np.float32) * 0.1
    return SharedSampler(from_list([Sample(id=f"a{i}", reference=r, audio=audio, sampling_rate=16000) for i, r in enumerate(REFS)]))


def test_ocr_episode():
    env = TeluguOCREnvironment(ocr_sampler())
    obs = env.reset()
    assert not obs.done and obs.reference is None and obs.reward is None
    img = Image.open(io.BytesIO(base64.b64decode(obs.image_png_b64)))
    assert img.size == (64, 32)
    obs = env.step(TranscriptionAction(transcription=REFS[0]))
    assert obs.done and obs.reward == 1.0 and obs.reference == REFS[0]
    assert env.state.step_count == 1 and env.state.sample_id == "s0"


def test_step_requires_reset_and_is_single_turn():
    env = TeluguOCREnvironment(ocr_sampler())
    assert env.step(TranscriptionAction(transcription="x")).error
    env.reset()
    env.step(TranscriptionAction(transcription="x"))
    assert env.step(TranscriptionAction(transcription=REFS[0])).error


def test_shared_sampler_and_group_replay():
    sampler = ocr_sampler()
    a, b = TeluguOCREnvironment(sampler), TeluguOCREnvironment(sampler)
    first = a.reset()
    assert b.reset().sample_id != first.sample_id  # concurrent sessions share one stream
    replay = b.reset(sample_id=first.sample_id)
    assert replay.image_png_b64 == first.image_png_b64
    with pytest.raises(KeyError):
        b.reset(sample_id="never-served")


def test_seeded_order_is_reproducible_and_epochs_wrap():
    ids = [TeluguOCREnvironment(ocr_sampler()).reset(seed=7).sample_id for _ in range(2)]
    assert ids[0] == ids[1]
    env = TeluguOCREnvironment(ocr_sampler())
    seen = [env.reset().sample_id for _ in range(len(REFS) + 1)]
    assert seen[-1] == seen[0]


def test_asr_episode_ignores_punctuation():
    env = TeluguASREnvironment(asr_sampler())
    obs = env.reset()
    audio, sr = sf.read(io.BytesIO(base64.b64decode(obs.audio_wav_b64)))
    assert sr == 16000 and len(audio) == 16000
    assert env.step(TranscriptionAction(transcription="తెలుగు, భాష!")).reward == 1.0


def test_websocket_server_roundtrip(monkeypatch):
    from fastapi.testclient import TestClient
    from openenv.core.env_server.http_server import create_app

    from telugu_envs.base import TranscriptionObservation

    sampler = ocr_sampler()
    app = create_app(lambda: TeluguOCREnvironment(sampler), TranscriptionAction, TranscriptionObservation, env_name="t", max_concurrent_envs=4)
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {}})
        reset = ws.receive_json()
        assert reset["data"]["observation"]["image_png_b64"]
        ws.send_json({"type": "step", "data": {"transcription": REFS[0]}})
        step = ws.receive_json()
        assert step["data"]["reward"] == 1.0 and step["data"]["done"] is True
