"""Transliteration scoring: character error rate plus an exact-match bonus.

Telugu words are long and inflected, so a word-level rate would score one wrong vowel
sign as a total miss. The reward is continuous in the character error rate, and a slice
pays for getting the whole word right, the same shape as the FineEnvs OCR and ASR
environments.
"""

from rapidfuzz.distance import Levenshtein

from ..text import clean_model_output, degeneracy_reasons, normalize

ERROR_WEIGHT = 0.8
EXACT_WEIGHT = 0.2
GRADING_POLICY = "te-translit-cer-v1"


def _rate(reference, prediction):
    if not reference:
        raise ValueError("Reference is empty after normalization; cannot score")
    # Capped at 1: inserting freely must not push the reward below zero.
    return min(1.0, Levenshtein.distance(reference, prediction) / len(reference))


def score(task, answer):
    """Grade one answer, returning the reward and the metrics behind it."""
    prediction = clean_model_output(answer)
    reference = normalize(task["reference"])
    guess = normalize(prediction)
    cer = _rate(reference, guess)
    wer = _rate(reference.split(), guess.split())
    exact = guess == reference
    empty = not guess
    # A looping or exploding answer scores 0, so GRPO can't farm partial credit by repeating.
    degenerate = bool(degeneracy_reasons(task["reference"], prediction))
    reward = 0.0 if empty or degenerate else ERROR_WEIGHT * (1.0 - cer) + EXACT_WEIGHT * float(exact)
    return round(reward, 6), {
        "cer": cer,
        "wer": wer,
        "exact_match": exact,
        "empty": empty,
        "degenerate": degenerate,
    }
