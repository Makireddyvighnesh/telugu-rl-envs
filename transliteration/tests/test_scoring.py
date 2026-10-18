"""The Space must score exactly like the repo's in-process reward used for the baselines."""

import random

import pytest

from telugu_transliteration.server.rewards import score

text_reward = pytest.importorskip("telugu_envs.reward").text_reward

PAIRS = [("తెలుగు", "telugu"), ("నమస్తే", "namaste"), ("హైదరాబాద్", "hyderabad"), ("శాక్రమెంటో", "sacramento"),
         ("సెట్టింగ్‌ల", "settingla"), ("౧౨ గంటలు", "12 gantalu")]


def answers(reference, rng):
    yield reference
    yield reference[:-1]
    yield reference + reference[-1]
    yield reference[::-1]
    yield ""
    yield f"<think>hmm</think>{reference}"
    yield f"```\n{reference}\n```"
    yield "Here is the text: " + reference
    yield (reference + " ") * 25
    chars = list(reference)
    rng.shuffle(chars)
    yield "".join(chars)


def test_matches_telugu_envs_text_reward():
    rng = random.Random(0)
    checked = 0
    for reference, _ in PAIRS:
        for answer in answers(reference, rng):
            reward, metrics = score({"reference": reference}, answer)
            expected = text_reward(reference, answer, "transliteration")
            assert reward == expected.reward, (reference, answer, reward, expected)
            assert metrics["cer"] == expected.cer and metrics["exact_match"] == expected.exact
            checked += 1
    assert checked == len(PAIRS) * 10
