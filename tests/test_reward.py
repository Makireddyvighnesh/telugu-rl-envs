import pytest

from telugu_envs.reward import cer, transcription_reward, wer
from telugu_envs.text import clean_model_output, normalize

REF = "తెలుగు భాష చాలా అందమైనది."


def test_exact_match_scores_one():
    r = transcription_reward(REF, REF)
    assert r.reward == 1.0 and r.cer == 0.0 and r.wer == 0.0 and not r.flags


def test_nfc_and_zero_width_are_ignored():
    decomposed = "ై"  # ె + ై-length-mark -> composes to ై (U+0C48) under NFC
    assert normalize(decomposed) == "ై"
    assert cer("తెలుగు", "తె‌లుగు") == 0.0


def test_telugu_digits_fold_to_ascii():
    assert cer("సంవత్సరం 2024", "సంవత్సరం ౨౦౨౪") == 0.0


def test_quotes_dashes_and_pipe_danda():
    assert cer("“నమస్తే” — అన్నాడు।", '"నమస్తే" - అన్నాడు|') == 0.0


def test_asr_mode_drops_punctuation():
    assert wer("నమస్తే, మిత్రమా!", "నమస్తే మిత్రమా", mode="asr") == 0.0
    assert wer("నమస్తే, మిత్రమా!", "నమస్తే మిత్రమా", mode="ocr") > 0.0


def test_partial_error_gives_partial_reward():
    r = transcription_reward(REF, "తెలుగు భాష చాల అందమైనది.")
    assert 0.9 < r.reward < 1.0
    assert r.wer == pytest.approx(0.25)


def test_wrappers_are_stripped():
    raw = "<think>let me read it</think>\n```\n" + REF + "\n```"
    assert clean_model_output(raw) == REF
    assert transcription_reward(REF, raw).reward == 1.0


def test_preamble_is_penalized():
    assert transcription_reward(REF, "Here is the text: " + REF).reward < 0.6


def test_empty_and_loops_get_zero():
    assert transcription_reward(REF, "").flags == ["empty"]
    assert transcription_reward(REF, "  ").reward == 0.0
    looped = REF + " అందమైనది" * 10
    r = transcription_reward(REF, looped)
    assert r.reward == 0.0 and "tail_loop" in r.flags


def test_rates_are_capped():
    assert cer("అ", "అఆఇఈఉఊ") == 1.0
    assert transcription_reward("అ", "ఆఇఈఉఊ").reward == 0.0
