"""Real-data checks against the Hugging Face Hub. Slow; opt in with RUN_HF_TESTS=1."""

import itertools
import os
import re

import pytest

from telugu_envs.base import SharedSampler, TranscriptionAction
from telugu_envs.data import SOURCES

pytestmark = pytest.mark.skipif(os.environ.get("RUN_HF_TESTS") != "1", reason="set RUN_HF_TESTS=1")

TELUGU = re.compile(r"[ఀ-౿]")


@pytest.mark.parametrize(
    "name,kwargs",
    [("nayana_regions", {}), ("nayana_natural", {}), ("sarvam", {}), ("fleurs", {"split": "validation"})],
)
def test_source_yields_valid_samples(name, kwargs):
    samples = list(itertools.islice(SOURCES[name](**kwargs)(0), 5))
    assert len(samples) == 5
    assert len({s.id for s in samples}) == 5
    for s in samples:
        assert s.reference.strip() and TELUGU.search(s.reference)
        if name == "fleurs":
            assert s.sampling_rate == 16000 and 0 < len(s.audio) / 16000 <= 30
        else:
            assert min(s.image.size) >= 16
            if name != "sarvam":  # benchmark images stay at native resolution
                assert max(s.image.size) <= 1536


def test_ocr_env_on_real_data_rewards_reference():
    from telugu_envs.ocr.environment import TeluguOCREnvironment

    sampler = SharedSampler(SOURCES["sarvam"]())
    env = TeluguOCREnvironment(sampler)
    obs = env.reset(seed=0)
    ref = sampler.get(obs.sample_id).reference
    assert env.step(TranscriptionAction(transcription=ref)).reward == 1.0
