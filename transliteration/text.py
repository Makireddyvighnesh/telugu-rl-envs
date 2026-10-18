"""Telugu text normalization and model-output cleanup (vendored from telugu_envs.text;
tests/test_scoring.py checks the two stay identical).

Normalization follows the content-normalization rules of the Sarvam Indic OCR
Bench scorer (NFKC/NFC, quote/dash folding, danda/pipe unification, ZWJ/ZWNJ,
markup and bullet stripping) so environment rewards line up with published
benchmark numbers. On top of that:

* Telugu digits (౦-౯) are folded to ASCII so "౧౨" and "12" score the same.
* ``mode="asr"`` additionally drops all punctuation, matching FLEURS'
  normalized ``transcription`` column.
"""

from __future__ import annotations

import re
import unicodedata

TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")

_ZERO_WIDTH_RE = re.compile(r"[​-‍⁠﻿]")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\b[^>]*>")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_CODE_FENCE_RE = re.compile(r"^```[\w-]*\s*\n?(.*?)\n?```\s*$", re.S)
_BULLETS = "•▪▫●○◦·‣⁃∙◆◇♦▸►▹➢➣➤→■□▣❑❖✦★☐"
_BULLET_RE = re.compile(rf"[{re.escape(_BULLETS)}]+")
_LIST_MARKER_RE = re.compile(r"(?:^|\s)(?:-|\d+[.):])(?=\s)")
_FILLER_RUN_RE = re.compile(r"([\-._=~─━═])\1+")
_LONG_RUN_RE = re.compile(r"(.)\1{4,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")


def clean_model_output(text: str) -> str:
    """Strip wrappers that VLM/ALM outputs commonly add around the answer.

    Removes ``<think>`` blocks (and anything before a dangling ``</think>``)
    and a single enclosing Markdown code fence. Prose like "Here is the text:"
    is deliberately *not* stripped: the reward should teach the policy to
    output only the transcription.
    """
    text = text or ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1)
    return text.strip()


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith(("P", "S"))


def normalize(text: str, mode: str = "ocr") -> str:
    """Normalize Telugu text for CER/WER scoring.

    Args:
        text: raw reference or prediction.
        mode: ``"ocr"`` keeps punctuation (folded to canonical forms);
            ``"asr"`` removes all punctuation and symbols.
    """
    if mode not in ("ocr", "asr"):
        raise ValueError(f"unknown mode {mode!r}")
    text = unicodedata.normalize("NFKC", text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(TELUGU_DIGITS)
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = _HTML_TAG_RE.sub(" ", text)
    text = _ZERO_WIDTH_RE.sub("", text)

    # Quotes, dashes, asterisks, danda.
    text = re.sub(r"[\"'“”‘’`´«»]", "", text)
    text = re.sub(r"[*∗✱⁎⋆＊]", "", text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace("||", "॥").replace("|", "।")

    text = _BULLET_RE.sub(" ", text)
    text = _FILLER_RUN_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub(" ", text)
    text = _LONG_RUN_RE.sub(lambda m: m.group(1) * 3, text)

    if mode == "asr":
        text = "".join(" " if _is_punct(ch) else ch for ch in text)
        text = text.lower()
    else:
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)

    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Degenerate-output detection (runaway repetition / length explosion)
# ---------------------------------------------------------------------------


def _tail_word_loop(words: list[str], min_repeats: int = 4) -> bool:
    for n in (1, 2, 3, 5):
        if len(words) < n * min_repeats:
            continue
        tail, reps, i = words[-n:], 0, len(words)
        while i >= n and words[i - n : i] == tail:
            reps += 1
            i -= n
        if reps >= min_repeats:
            return True
    return False


def _tail_block_loop(text: str, min_block: int = 12, min_repeats: int = 3, max_block: int = 500) -> bool:
    n = len(text)
    for size in range(min_block, min(max_block, n // min_repeats) + 1):
        if text[-size * min_repeats :] == text[-size:] * min_repeats:
            return True
    return False


def degeneracy_reasons(reference: str, prediction: str) -> list[str]:
    """Return why ``prediction`` looks degenerate (empty list if it doesn't)."""
    reasons = []
    words = prediction.split()
    ref_words = reference.split()
    if _tail_word_loop(words) and not _tail_word_loop(ref_words):
        reasons.append("tail_loop")
    elif _tail_block_loop(prediction) and not _tail_block_loop(reference):
        reasons.append("tail_loop")
    if ref_words and len(words) >= 2 * len(ref_words) and len(words) >= 20:
        reasons.append("length_explosion")
    return reasons
