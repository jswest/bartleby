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
    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT,
    )
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
