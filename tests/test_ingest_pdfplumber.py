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
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

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
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

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
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

    assert result.page_count == 1
    page = result.pages[0]
    assert page.is_sparse
    # OCR on a page with just "abc" cannot clear the 100-char threshold, so
    # content_type=None and text="" — the caller will route the page render
    # through the VLM instead.
    assert page.content_type is None
    assert page.text == ""
    assert page.page_render_png is not None
    assert page.page_render_png.startswith(b"\x89PNG")


def test_convert_does_not_render_text_pages_without_images(tmp_path):
    # Regression: rendering every text-only page is wasteful. Pages that have
    # neither sparse text nor embedded images should skip rendering entirely.
    src = tmp_path / "text.pdf"
    _text_pdf(src, [("Plenty of text to clear the sparse threshold easily. " * 4)])
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)
    assert result.pages[0].page_render_png is None
    assert result.pages[0].embedded_images == []


def test_convert_skips_page_substrate_image(tmp_path):
    """OCR'd-scan PDFs embed a full-page raster under an OCR text overlay.
    That raster's content is already covered by extract_text(), so the
    cropper should drop it instead of routing it through the VLM."""
    src = tmp_path / "ocr_overlay.pdf"
    c = canvas.Canvas(str(src), pagesize=letter)
    page_w, page_h = letter
    # First the substrate: a page-sized image underneath everything.
    substrate = _png_image(width=int(page_w), height=int(page_h), color=(220, 220, 220))
    buf = io.BytesIO()
    substrate.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), 0, 0, width=page_w, height=page_h)
    # Then a small "real" image on top of it (a signature box, say).
    sig = _png_image(width=120, height=40, color=(50, 50, 200))
    sbuf = io.BytesIO()
    sig.save(sbuf, format="PNG")
    sbuf.seek(0)
    c.drawImage(ImageReader(sbuf), 72, 100, width=120, height=40)
    # And real text so pdfplumber extracts the page (matches the OCR-overlay shape).
    c.setFont("Helvetica", 12)
    t = c.beginText(72, 700)
    for _ in range(8):
        t.textLine("Plenty of selectable text on this page from the OCR overlay.")
    c.drawText(t)
    c.showPage()
    c.save()

    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)
    page = result.pages[0]
    # The substrate is dropped; only the small signature crop survives.
    assert len(page.embedded_images) == 1


def test_convert_skips_subpixel_embedded_images(tmp_path):
    """Sub-pixel-tall bbox would truncate to 0 height after PIL crop and
    crash JPEG encoding with 'cannot write empty image'. Skip them upfront."""
    src = tmp_path / "subpixel.pdf"
    c = canvas.Canvas(str(src), pagesize=letter)
    page_w, page_h = letter
    # A normal-sized embedded image, plus an extremely thin one (0.5pt tall)
    # that pdfplumber will register as a separate image entry.
    normal = _png_image(width=200, height=100)
    nbuf = io.BytesIO(); normal.save(nbuf, format="PNG"); nbuf.seek(0)
    c.drawImage(ImageReader(nbuf), 100, 400, width=200, height=100)
    thin = _png_image(width=400, height=2)
    tbuf = io.BytesIO(); thin.save(tbuf, format="PNG"); tbuf.seek(0)
    c.drawImage(ImageReader(tbuf), 100, 350, width=400, height=0.4)
    c.setFont("Helvetica", 12)
    t = c.beginText(72, 700)
    for _ in range(6):
        t.textLine("Enough text to keep the page out of the sparse path.")
    c.drawText(t)
    c.showPage()
    c.save()

    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)
    page = result.pages[0]
    # Only the normal image survives — the sub-pixel thin one is skipped.
    assert len(page.embedded_images) == 1
    # And the surviving crop is real — Pillow can open it and it has >0 area.
    decoded = Image.open(io.BytesIO(page.embedded_images[0].png_bytes))
    assert decoded.size[0] > 0 and decoded.size[1] > 0


def test_convert_calls_on_progress_with_total_then_each_page(tmp_path):
    src = tmp_path / "multi.pdf"
    _text_pdf(src, [
        ("Page one body. " * 12),
        ("Page two body. " * 12),
        ("Page three body. " * 12),
    ])
    seen: list[tuple[int, int]] = []
    pp.convert(
        src,
        sparse_text_threshold=100,
        ocr_min_confidence=30,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    # First call signals the total; later calls report each page.
    assert seen[0] == (0, 3)
    assert seen[1:] == [(1, 3), (2, 3), (3, 3)]
