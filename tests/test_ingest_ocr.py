"""Unit tests for bartleby.ingest.ocr.

These exercise Tesseract via pytesseract — the binary must be installed
locally. Tests render text with PIL/freetype-via-PIL so we don't depend on
specific fonts being present; they're tolerant of OCR variance.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from bartleby.ingest import ocr


def _render_text(text: str, size=(600, 200)) -> bytes:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    # Try a couple of common system font paths; fall back to the bitmap default.
    font = None
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(path, 48)
            break
        except OSError:
            continue
    draw.text((20, 60), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _blank_image() -> bytes:
    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def _check_tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:
        pytest.skip(f"Tesseract not available: {e}")


def test_run_reads_clear_text(_check_tesseract):
    png = _render_text("HELLO WORLD")
    result = ocr.run(png)
    assert "HELLO" in result.text.upper()
    assert "WORLD" in result.text.upper()
    assert result.avg_confidence > 30


def test_run_on_blank_returns_no_text(_check_tesseract):
    png = _blank_image()
    result = ocr.run(png)
    assert result.text == ""
    # No words → sentinel confidence.
    assert result.avg_confidence == -1.0


def test_run_surfaces_legible_error_on_decode_crash(monkeypatch):
    """When tesseract emits non-UTF-8 stderr (e.g. it can't read its temp file),
    pytesseract raises a bare UnicodeDecodeError. `run` must convert it into a
    legible RuntimeError that names the likely TMPDIR cause (issue #43)."""
    def _boom(*a, **k):
        raise UnicodeDecodeError("utf-8", b"\x89PNG", 0, 1, "invalid start byte")
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        ocr.run(_blank_image())
    msg = str(excinfo.value)
    assert "Tesseract OCR failed" in msg
    assert "TMPDIR" in msg
    # The original cause is preserved for debugging.
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_run_surfaces_legible_error_on_tesseract_error(monkeypatch):
    """A normal TesseractError is likewise wrapped into the legible RuntimeError."""
    def _boom(*a, **k):
        raise ocr.pytesseract.TesseractError(1, "tesseract exploded")
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        ocr.run(_blank_image())
    assert "Tesseract OCR failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ocr.pytesseract.TesseractError)


def test_run_surfaces_legible_error_when_binary_missing(monkeypatch):
    """A missing tesseract binary raises TesseractNotFoundError — the most common
    #43 breakage. `run` must wrap it into the legible RuntimeError too, so callers
    (e.g. pdfplumber's sparse-page classifier) see the actionable cause (#309)."""
    def _boom(*a, **k):
        raise ocr.pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        ocr.run(_blank_image())
    assert "Tesseract OCR failed" in str(excinfo.value)
    assert isinstance(
        excinfo.value.__cause__, ocr.pytesseract.TesseractNotFoundError
    )
