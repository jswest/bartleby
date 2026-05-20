"""Unit tests for bartleby.ingest.pdfplumber.

Generates tiny PDFs on disk with reportlab so the tests are hermetic.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from bartleby.ingest import pdfplumber as pp


def _text_pdf(path, pages: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    for body in pages:
        c.setFont("Helvetica", 12)
        text = c.beginText(72, 720)
        for line in body.splitlines():
            text.textLine(line)
        c.drawText(text)
        c.showPage()
    c.save()


def _png_image(width=200, height=100, color=(0, 128, 255)):
    return Image.new("RGB", (width, height), color=color)


def _pdf_with_embedded_image(path, image: Image.Image) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 12)
    text = c.beginText(72, 720)
    for _ in range(6):
        text.textLine("This page has plenty of text to clear the sparse threshold.")
    c.drawText(text)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), 100, 400, width=200, height=100)
    c.showPage()
    c.save()


def _sparse_pdf(path) -> None:
    # A page with just three characters — well under the default 100 threshold.
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "abc")
    c.showPage()
    c.save()


def test_convert_text_only_pdf(tmp_path):
    src = tmp_path / "text.pdf"
    _text_pdf(src, [
        ("This is page one of a small PDF. " * 6),
        ("Page two also has text. " * 8),
    ])
    result = pp.convert(src, sparse_text_threshold=100)

    assert result.page_count == 2
    assert len(result.pages) == 2
    assert "page one" in result.pages[0].text.lower()
    assert "page two" in result.pages[1].text.lower()
    assert all(not p.is_sparse for p in result.pages)
    assert all(p.page_render_png is None for p in result.pages)
    assert all(p.embedded_images == [] for p in result.pages)
    # full_text concatenates everything.
    assert "page one" in result.full_text.lower()
    assert "page two" in result.full_text.lower()


def test_convert_extracts_embedded_image(tmp_path):
    src = tmp_path / "withimg.pdf"
    _pdf_with_embedded_image(src, _png_image())
    result = pp.convert(src, sparse_text_threshold=100)

    assert result.page_count == 1
    page = result.pages[0]
    assert not page.is_sparse
    # At least one embedded image is detected and cropped to PNG bytes.
    assert len(page.embedded_images) >= 1
    img = page.embedded_images[0]
    assert img.image_index_on_page == 1
    assert img.png_bytes.startswith(b"\x89PNG")
    # The crop is a real image — Pillow can open it.
    decoded = Image.open(io.BytesIO(img.png_bytes))
    assert decoded.size[0] > 0 and decoded.size[1] > 0


def test_convert_marks_sparse_pages_and_saves_render(tmp_path):
    src = tmp_path / "sparse.pdf"
    _sparse_pdf(src)
    result = pp.convert(src, sparse_text_threshold=100)

    assert result.page_count == 1
    page = result.pages[0]
    assert page.is_sparse
    assert page.text == "abc"
    assert page.page_render_png is not None
    assert page.page_render_png.startswith(b"\x89PNG")


def test_convert_does_not_render_text_pages_without_images(tmp_path):
    # Regression: rendering every text-only page is wasteful. Pages that have
    # neither sparse text nor embedded images should skip rendering entirely.
    src = tmp_path / "text.pdf"
    _text_pdf(src, [("Plenty of text to clear the sparse threshold easily. " * 4)])
    result = pp.convert(src, sparse_text_threshold=100)
    assert result.pages[0].page_render_png is None
    assert result.pages[0].embedded_images == []
