"""Tesseract OCR wrapper.

Used as the cheap first pass for sparse PDF pages: if Tesseract finds
sufficient text at sufficient confidence, we treat the page as text
(``content_type='ocr'``, ``source_kind='document'``). Otherwise the caller
falls back to the VLM via the image pipeline.

No image preprocessing — Tesseract's built-in pipeline handles the basics,
and our renderer (pypdfium2 at 150 DPI) already produces clean inputs.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

from PIL import Image
import pytesseract


@dataclass
class OcrResult:
    text: str
    avg_confidence: float   # 0-100; -1.0 when Tesseract found nothing


def run(image_bytes: bytes) -> OcrResult:
    """Run Tesseract on a PNG/JPEG. Returns the recognized text + avg confidence."""
    img = Image.open(io.BytesIO(image_bytes))
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT,
        )
    except (
        pytesseract.TesseractError,
        pytesseract.TesseractNotFoundError,
        UnicodeDecodeError,
    ) as e:
        # pytesseract writes the image to a temp file under $TMPDIR and shells
        # out to `tesseract`. When that subprocess can't read the temp file,
        # tesseract emits a non-UTF-8 diagnostic and pytesseract crashes
        # decoding it with a UnicodeDecodeError that buries the real cause.
        # A missing binary raises TesseractNotFoundError — the most common #43
        # breakage — which we likewise re-wrap so callers see the legible cause.
        # Re-raise something legible and point at the usual culprit. See #43.
        raise RuntimeError(
            f"Tesseract OCR failed ({type(e).__name__}: {e}). The tesseract "
            f"subprocess may be unable to read its temp file under "
            f"TMPDIR={os.environ.get('TMPDIR', '')!r}."
        ) from e
    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data["text"], data["conf"]):
        if not word or not word.strip():
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        words.append(word)
        confidences.append(c)

    text = " ".join(words).strip()
    avg = sum(confidences) / len(confidences) if confidences else -1.0
    return OcrResult(text=text, avg_confidence=avg)
