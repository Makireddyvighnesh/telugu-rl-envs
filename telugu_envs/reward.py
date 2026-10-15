"""CER/WER-based transcription reward shared by the OCR and ASR environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rapidfuzz.distance import Levenshtein

from telugu_envs.text import clean_model_output, degeneracy_reasons, normalize


def _rate(ref: list | str, hyp: list | str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(1.0, Levenshtein.distance(ref, hyp) / len(ref))


def cer(reference: str, prediction: str, mode: str = "ocr") -> float:
    """Character error rate over Unicode code points, capped at 1.0."""
    return _rate(normalize(reference, mode), normalize(prediction, mode))


def wer(reference: str, prediction: str, mode: str = "ocr") -> float:
    """Word error rate over whitespace tokens, capped at 1.0."""
    return _rate(normalize(reference, mode).split(), normalize(prediction, mode).split())


@dataclass
class RewardResult:
    reward: float
    cer: float
    wer: float
    prediction: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def chrf(reference: str, prediction: str) -> float:
    """chrF++ (character n-grams + word bigrams) in [0, 1]; standard MT metric."""
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2).sentence_score(normalize(prediction), [normalize(reference)]).score / 100.0


@dataclass
class TextRewardResult:
    reward: float
    cer: float
    chrf: float
    exact: bool
    prediction: str
    flags: list[str] = field(default_factory=list)


def text_reward(reference: str, raw_prediction: str, task: str) -> TextRewardResult:
    """Reward for text-to-text tasks.

    transliteration: one right answer, so 0.8 * (1 - CER) + 0.2 * exact match
        (the exact-match bonus is the FineEnvs recipe: it rewards getting the
        whole word right, not just most characters).
    translation: many right answers, so chrF++ (overlap with one reference
        translation). A known weakness: a valid paraphrase scores lower than it should.
    """
    prediction = clean_model_output(raw_prediction)
    c = cer(reference, prediction)
    exact = normalize(prediction) == normalize(reference)
    ch = chrf(reference, prediction)
    flags = []
    if not normalize(prediction):
        flags.append("empty")
    flags += degeneracy_reasons(reference, prediction)
    if flags:
        reward = 0.0
    elif task == "transliteration":
        reward = 0.8 * (1.0 - c) + 0.2 * exact
    elif task == "translation":
        reward = ch
    else:
        raise ValueError(f"unknown task {task!r}")
    return TextRewardResult(reward=round(reward, 6), cer=c, chrf=ch, exact=exact, prediction=prediction, flags=flags)


def transcription_reward(
    reference: str,
    raw_prediction: str,
    mode: str = "ocr",
    cer_weight: float = 1.0,
) -> RewardResult:
    """Score a transcription.

    reward = 1 - (cer_weight * CER + (1 - cer_weight) * WER), in [0, 1].
    Empty and degenerate (looping / exploding) outputs get reward 0 regardless
    of their error rate, so the policy can't farm partial credit by repeating.
    """
    prediction = clean_model_output(raw_prediction)
    c = cer(reference, prediction, mode)
    w = wer(reference, prediction, mode)
    flags = []
    if not normalize(prediction, mode):
        flags.append("empty")
    flags += degeneracy_reasons(reference, prediction)
    reward = 0.0 if flags else 1.0 - (cer_weight * c + (1.0 - cer_weight) * w)
    return RewardResult(reward=round(reward, 6), cer=c, wer=w, prediction=prediction, flags=flags)
