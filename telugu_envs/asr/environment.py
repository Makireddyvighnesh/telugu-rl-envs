"""Telugu ASR environment: speech clip -> transcription, rewarded by 1 - CER.

Scored in ``asr`` mode (punctuation ignored), matching FLEURS' normalized
references. Use ``cer_weight<1`` to blend in WER.
"""

from __future__ import annotations

from telugu_envs.base import SharedSampler, TranscriptionEnvironment, encode_audio
from telugu_envs.data import Sample

ASR_INSTRUCTION = (
    "Transcribe this Telugu speech in Telugu script. "
    "Output only the transcription, with no explanation or translation."
)


class TeluguASREnvironment(TranscriptionEnvironment):
    MODE = "asr"
    INSTRUCTION = ASR_INSTRUCTION

    def __init__(self, sampler: SharedSampler, **kwargs):
        super().__init__(sampler, **kwargs)

    def _encode(self, sample: Sample) -> dict:
        return {"audio_wav_b64": encode_audio(sample.audio, sample.sampling_rate), "sampling_rate": sample.sampling_rate}
