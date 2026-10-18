import pytest

from telugu_transliteration.models import TransliterationAction
from telugu_transliteration.server.environment import TransliterationEnvironment


def test_episode_scores_once_and_never_leaks_reference(catalog):
    env = TransliterationEnvironment(catalog)
    obs = env.reset(task_id="te-translit/train/fx0")
    assert not obs.done and obs.reward is None and obs.source_text == "telugu"
    assert "telugu" in obs.prompt and "reference" not in obs.model_dump()
    obs = env.step(TransliterationAction(answer="తెలుగు"))
    assert obs.done and obs.reward == 1.0 and obs.metrics["exact_match"] is True
    with pytest.raises(ValueError):
        env.step(TransliterationAction(answer="తెలుగు"))


def test_reset_by_split_index_and_seed(catalog):
    env = TransliterationEnvironment(catalog)
    assert env.reset(split="test", index=2).task_id == "te-translit/test/fx2"
    assert env.reset(seed=7).task_id == env.reset(seed=7).task_id
    with pytest.raises(ValueError):
        env.reset(task_id="te-translit/train/fx0", split="train")


def test_task_api_is_public_only(catalog):
    env = TransliterationEnvironment(catalog)
    assert set(env.list_splits()) == {"train", "validation", "test", "validation-256"}
    assert env.num_tasks("validation-256") == 2
    tasks = env.get_task_range("train", 1, 3)
    assert [t["index"] for t in tasks] == [1, 2]
    assert all("reference" not in t for t in tasks)


@pytest.mark.parametrize(
    "answer,check",
    [
        ("", lambda r, m: r == 0.0 and m["empty"]),
        ("<think>x</think>తెలుగు", lambda r, m: r == 1.0),
        ("తెలుగ", lambda r, m: 0 < r < 0.8 and not m["exact_match"]),
        ("తెలుగు " * 30, lambda r, m: r == 0.0 and m["degenerate"]),
    ],
)
def test_reward_paths(catalog, answer, check):
    env = TransliterationEnvironment(catalog)
    env.reset(task_id="te-translit/train/fx0")
    obs = env.step(TransliterationAction(answer=answer))
    assert check(obs.reward, obs.metrics), (answer, obs.reward, obs.metrics)
