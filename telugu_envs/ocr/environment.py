"""Telugu OCR environment: image of Telugu text -> transcription, rewarded by 1 - CER."""

from __future__ import annotations

from telugu_envs.base import SharedSampler, TranscriptionEnvironment, encode_image
from telugu_envs.data import Sample

OCR_INSTRUCTION = (
    "Transcribe all the Telugu text in this image exactly as written, preserving "
    "punctuation and numbers. Output only the transcribed text, with no explanation."
)


class TeluguOCREnvironment(TranscriptionEnvironment):
    MODE = "ocr"
    INSTRUCTION = OCR_INSTRUCTION

    def __init__(self, sampler: SharedSampler, **kwargs):
        super().__init__(sampler, **kwargs)

    def _encode(self, sample: Sample) -> dict:
        return {"image_png_b64": encode_image(sample.image)}
