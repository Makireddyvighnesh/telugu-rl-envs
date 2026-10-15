"""Streaming sample sources for the Telugu OCR and ASR environments.

Every source is an iterable of :class:`Sample`. Hugging Face sources stream
(``load_dataset(..., streaming=True)``) so nothing large is downloaded up front;
``seed`` controls the shuffle-buffer order so episodes are reproducible.

Sources
-------
OCR train  : ``nayana_regions``   Cognitive-Lab/NayanaOCR_Corpus_2025 [te], synthetic
             pages cropped to single layout regions (exact rendered text as GT).
OCR eval   : ``nayana_natural``   Cognitive-Lab/NayanaOCRBench_Natural [te], 461 real,
             human-verified pages cropped to text blocks.
OCR test   : ``sarvam``           sarvamai/indic-ocr-bench, Telugu block crops (300).
ASR        : ``fleurs``           google/fleurs [te_in] train / validation / test.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Nayana pages are 3300x2550; Natural scans can be larger.

_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class Sample:
    id: str
    reference: str
    image: Image.Image | None = None
    audio: np.ndarray | None = None
    sampling_rate: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


SampleSource = Callable[[int | None], Iterator[Sample]]
"""A callable ``seed -> iterator of samples``; ``seed=None`` means dataset order."""


def _stream(repo: str, split: str, seed: int | None, buffer: int, **kwargs):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split, streaming=True, **kwargs)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=buffer)
    return ds


def _to_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and img.get("bytes"):
        return Image.open(io.BytesIO(img["bytes"]))
    raise TypeError(f"cannot decode image of type {type(img)}")


def _crop(page: Image.Image, box: tuple[float, float, float, float], pad: int, max_side: int) -> Image.Image:
    x0, y0, x1, y1 = box
    W, H = page.size
    crop = page.crop((max(0, int(x0) - pad), max(0, int(y0) - pad), min(W, int(x1) + pad), min(H, int(y1) + pad)))
    crop = crop.convert("RGB")
    if max(crop.size) > max_side:
        crop.thumbnail((max_side, max_side), Image.LANCZOS)
    return crop


# ---------------------------------------------------------------------------
# OCR sources
# ---------------------------------------------------------------------------

TEXT_LAYOUTS = ("text", "title", "list", "caption", "header", "footer")


def nayana_regions(
    min_chars: int = 8,
    max_chars: int = 400,
    min_height: int = 24,
    pad: int = 16,
    max_side: int = 1536,
    layouts: Iterable[str] = TEXT_LAYOUTS,
    require_telugu: bool = True,
    buffer: int = 64,
) -> SampleSource:
    """Single-region crops from Nayana's synthetic Telugu page renders."""
    layouts = set(layouts)

    def source(seed: int | None) -> Iterator[Sample]:
        rng = random.Random(seed)
        for row in _stream("Cognitive-Lab/NayanaOCR_Corpus_2025", "train", seed, buffer, data_dir="te"):
            page = None
            seen: set[str] = set()  # some pages carry duplicate region annotations
            regions = list(row["regions.json"] or [])
            if seed is not None:
                rng.shuffle(regions)
            for reg in regions:
                text = (reg.get("translated_text") or "").strip()
                b = reg["bbox"]
                if reg.get("layout_type") not in layouts or not (min_chars <= len(text) <= max_chars):
                    continue
                if require_telugu and not _TELUGU_RE.search(text):
                    continue
                if text in seen:
                    continue
                seen.add(text)
                if b["ymax"] - b["ymin"] < min_height or b["xmax"] - b["xmin"] < min_height:
                    continue
                page = page or _to_pil(row["jpg"])
                yield Sample(
                    id=f"nayana/{row['image_id.txt']}/r{reg['region_id']}",
                    reference=text,
                    image=_crop(page, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]), pad, max_side),
                    meta={"source": "nayana_regions", "layout": reg["layout_type"], "font": row.get("font_used.txt")},
                )

    return source


def nayana_natural(
    min_chars: int = 8,
    max_chars: int = 600,
    pad: int = 8,
    max_side: int = 1536,
    layouts: Iterable[str] = ("text", "title", "text_block", "paragraph", "list", "caption", "header", "footer"),
    require_telugu: bool = True,
    buffer: int = 16,
) -> SampleSource:
    """Text-block crops from human-verified real Telugu documents (eval set)."""
    layouts = {l.lower() for l in layouts}

    def source(seed: int | None) -> Iterator[Sample]:
        for row in _stream("Cognitive-Lab/NayanaOCRBench_Natural", "train", seed, buffer, data_dir="te"):
            page = None
            for i, blk in enumerate(row["layout_content_norm1000"] or []):
                text = (blk.get("content") or "").strip()
                if (blk.get("type") or "").lower() not in layouts or not (min_chars <= len(text) <= max_chars):
                    continue
                if require_telugu and not _TELUGU_RE.search(text):
                    continue
                page = page or _to_pil(row["image"])
                W, H = page.size
                x0, y0, x1, y1 = blk["bbox"]
                box = (x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000)
                yield Sample(
                    id=f"natural/{row['id']}/b{i}",
                    reference=text,
                    image=_crop(page, box, pad, max_side),
                    meta={"source": "nayana_natural", "layout": blk.get("type")},
                )

    return source


def sarvam(split: str = "test", max_side: int | None = None, buffer: int = 64) -> SampleSource:
    """Telugu blocks of the Sarvam Indic OCR Bench (held-out test set).

    Images are kept at native resolution (up to ~2000 px wide) to match the
    benchmark; pass ``max_side`` to downscale for models with small image budgets.
    """

    def source(seed: int | None) -> Iterator[Sample]:
        for row in _stream("sarvamai/indic-ocr-bench", split, seed, buffer):
            if row["language"] != "Telugu":
                continue
            image = _to_pil(row["image"]).convert("RGB")
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.LANCZOS)
            yield Sample(
                id=f"sarvam/{row['image_name']}",
                reference=row["gt"],
                image=image,
                meta={"source": "sarvam"},
            )

    return source


# ---------------------------------------------------------------------------
# ASR sources
# ---------------------------------------------------------------------------


def _decode_audio(data: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa

        arr, sr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr), target_sr
    return arr, sr


def fleurs(split: str = "train", max_seconds: float = 30.0, exclude_latin: bool = False, buffer: int = 64) -> SampleSource:
    """FLEURS Telugu (te_in). Reference is the normalized ``transcription``.

    ~10% of references keep English names/terms in Latin script ("qvc",
    "siminoff"); a model that transliterates them into Telugu is penalized.
    ``exclude_latin=True`` drops those utterances.
    """

    def source(seed: int | None) -> Iterator[Sample]:
        from datasets import Audio

        # google/fleurs still ships a loading script; read its parquet export directly.
        ds = _stream(
            "parquet",
            split,
            seed,
            buffer,
            data_files={split: f"hf://datasets/google/fleurs/parquet-data/te_in/{split}-*.parquet"},
        ).cast_column("audio", Audio(decode=False))  # decode with soundfile; avoids a torchcodec dependency
        for row in ds:
            if exclude_latin and _LATIN_RE.search(row["transcription"]):
                continue
            arr, sr = _decode_audio(row["audio"]["bytes"], target_sr=16000)
            if len(arr) / sr > max_seconds:
                continue
            yield Sample(
                id=f"fleurs/{split}/{row['id']}/{row['path'].rsplit('/', 1)[-1]}",
                reference=row["transcription"],
                audio=arr,
                sampling_rate=sr,
                meta={"source": "fleurs", "raw_transcription": row["raw_transcription"], "gender": row["gender"]},
            )

    return source


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def from_list(samples: list[Sample]) -> SampleSource:
    """In-memory source (tests, cached subsets). Seed shuffles a copy."""

    def source(seed: int | None) -> Iterator[Sample]:
        items = list(samples)
        if seed is not None:
            random.Random(seed).shuffle(items)
        return iter(items)

    return source


SOURCES: dict[str, Callable[..., SampleSource]] = {
    "nayana_regions": nayana_regions,
    "nayana_natural": nayana_natural,
    "sarvam": sarvam,
    "fleurs": fleurs,
}
