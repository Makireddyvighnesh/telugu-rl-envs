import pytest

from telugu_envs.base import SharedSampler
from telugu_envs.data import Sample, from_list
from telugu_envs.reward import text_reward
from telugu_envs.text_env import TeluguTextEnvironment, TextAction

TRANSLIT = Sample(id="t0", text="nenu intiki veltunnanu", reference="నేను ఇంటికి వెళ్తున్నాను",
                  meta={"task": "transliteration"})
TRANSLATE = Sample(id="m0", text="I am going home.", reference="నేను ఇంటికి వెళ్తున్నాను.",
                   meta={"task": "translation", "direction": "eng-tel"})


def test_transliteration_reward_shape():
    exact = text_reward(TRANSLIT.reference, TRANSLIT.reference, "transliteration")
    near = text_reward(TRANSLIT.reference, "నేను ఇంటికి వెల్తున్నాను", "transliteration")  # ళ -> ల
    assert exact.reward == 1.0 and exact.exact
    assert 0.7 < near.reward < 0.8  # loses the 0.2 exact-match bonus, keeps most of 0.8
    assert text_reward(TRANSLIT.reference, "nenu intiki veltunnanu", "transliteration").reward < 0.1


def test_translation_reward_is_chrf():
    exact = text_reward(TRANSLATE.reference, TRANSLATE.reference, "translation")
    para = text_reward(TRANSLATE.reference, "నేను ఇంటికి వెళ్తాను.", "translation")
    assert exact.reward == 1.0
    assert 0.3 < para.reward < 0.9  # partial credit for a close paraphrase
    with pytest.raises(ValueError):
        text_reward("a", "a", "summarization")


def test_text_env_episode_and_prompt():
    env = TeluguTextEnvironment(SharedSampler(from_list([TRANSLIT, TRANSLATE])))
    obs = env.reset()
    assert obs.task == "transliteration" and obs.source_text in obs.prompt and obs.reference is None
    assert env.step(TextAction(answer=TRANSLIT.reference)).reward == 1.0
    obs = env.reset()
    assert obs.task == "translation" and "into Telugu" in obs.prompt
    res = env.step(TextAction(answer="<think>hmm</think>" + TRANSLATE.reference))
    assert res.reward == 1.0 and res.done
