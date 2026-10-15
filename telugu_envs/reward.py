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
