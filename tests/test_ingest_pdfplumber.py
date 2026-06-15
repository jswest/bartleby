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
    assert all(p.content_type == "text" for p in result.pages)
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
    assert page.content_type == "text"
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
    # OCR on a page with just "abc" cannot clear the 100-char threshold, so
    # content_type=None and text="" — the caller will route the page render
    # through the VLM instead.
    assert page.content_type is None
    assert page.text == ""
    assert page.page_render_png is not None
    assert page.page_render_png.startswith(b"\x89PNG")


def _mixed_sparse_and_text_pdf(path) -> None:
    # Page 1 is sparse ("abc"); page 2 has plenty of text to clear the threshold.
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "abc")
    c.showPage()
    c.setFont("Helvetica", 12)
    text = c.beginText(72, 720)
    for _ in range(6):
        text.textLine("This page has plenty of text to clear the sparse threshold.")
    c.drawText(text)
    c.showPage()
    c.save()


def test_convert_degrades_to_vlm_when_ocr_raises(tmp_path, monkeypatch):
    # #309: a Tesseract failure during sparse-page classification (broken
    # install, locked TMPDIR) must NOT fail the whole PDF parse. The sparse page
    # degrades to the VLM route and every non-sparse page still chunks normally.
    src = tmp_path / "mixed.pdf"
    _mixed_sparse_and_text_pdf(src)

    def _boom(_image_bytes):
        raise RuntimeError("Tesseract OCR failed (TesseractNotFoundError).")

    monkeypatch.setattr(pp.ocr_module, "run", _boom)

    # The parse succeeds rather than propagating the OCR failure.
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

    assert result.page_count == 2
    sparse_page, text_page = result.pages

    # Sparse page: OCR raised → routed to the VLM (content_type None, no text),
    # with its render preserved so the caller can hand it to the VLM.
    assert sparse_page.content_type is None
    assert sparse_page.text == ""
    assert sparse_page.page_render_png is not None
    assert sparse_page.page_render_png.startswith(b"\x89PNG")

    # Non-sparse page is untouched by the OCR failure — still chunks as text.
    assert text_page.content_type == "text"
    assert "plenty of text" in text_page.text.lower()


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


def test_convert_skips_decorative_small_and_thin_images(tmp_path):
    """Filter decorative icons (too small on long edge) and thin strips
    (pass long-edge but tiny area). Real content survives."""
    src = tmp_path / "decorative.pdf"
    c = canvas.Canvas(str(src), pagesize=letter)
    # Real chart-sized image — should survive.
    real = _png_image(width=300, height=200, color=(10, 100, 200))
    rbuf = io.BytesIO(); real.save(rbuf, format="PNG"); rbuf.seek(0)
    c.drawImage(ImageReader(rbuf), 100, 500, width=300, height=200)
    # Small decorative icon (small in both dims) — skipped by long-edge rule.
    icon = _png_image(width=40, height=40, color=(0, 0, 0))
    ibuf = io.BytesIO(); icon.save(ibuf, format="PNG"); ibuf.seek(0)
    c.drawImage(ImageReader(ibuf), 100, 450, width=20, height=20)
    # Thin decorative bar (passes long edge, fails area) — skipped by area rule.
    # 400pt × 4pt at 150 DPI → ~833 × 8 px = 6,664 px²... wait need to ensure
    # this falls under 5000 px². Use 300pt × 1pt → ~625 × 2 px = 1,250 px².
    bar = _png_image(width=300, height=2, color=(128, 128, 128))
    bbuf = io.BytesIO(); bar.save(bbuf, format="PNG"); bbuf.seek(0)
    c.drawImage(ImageReader(bbuf), 100, 430, width=300, height=1)
    c.setFont("Helvetica", 12)
    t = c.beginText(72, 700)
    for _ in range(6):
        t.textLine("Plenty of text so the page is not sparse.")
    c.drawText(t)
    c.showPage()
    c.save()

    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)
    page = result.pages[0]
    # Only the real chart survives — the icon and thin bar are dropped.
    assert len(page.embedded_images) == 1


def test_reject_if_html_rejects_html_saved_as_pdf(tmp_path):
    # The exact NY DPS portal shape from #235: leading CRLF then an HTML page,
    # saved with a `.pdf` extension by a failed download.
    src = tmp_path / "ViewDoc.pdf"
    src.write_bytes(
        b"\r\n\r\n<!DOCTYPE html>\n<html><body>"
        b"Either the document does not exists or some problem occured."
        b"</body></html>"
    )
    with pytest.raises(pp.NotAPdfError) as exc:
        pp.reject_if_html(src)
    # Actionable reason, not pdfminer's cryptic "No /Root object!".
    assert "HTML page" in str(exc.value)
    assert "No /Root object" not in str(exc.value)


def test_reject_if_html_passes_a_real_pdf(tmp_path):
    src = tmp_path / "real.pdf"
    _text_pdf(src, ["A genuine PDF with plenty of text. " * 4])
    # A valid PDF starts with %PDF — no exception.
    pp.reject_if_html(src)


# ---------------------------------------------------------------------------
# Vector figure detection tests (#3)
# ---------------------------------------------------------------------------

def _pdf_with_vector_figure_and_raster(path, image: Image.Image) -> None:
    """One page: a vector chart (axes + diagonal lines + bezier) plus a raster
    image. The chart uses non-crossing lines so pdfplumber's table detector
    does not claim the cluster."""
    from reportlab.lib import colors as rl_colors
    c = canvas.Canvas(str(path), pagesize=letter)
    # Enough text so the page stays out of the sparse path.
    t = c.beginText(72, 750)
    for _ in range(4):
        t.textLine("Text to clear the sparse threshold and keep page content_type='text'.")
    c.drawText(t)
    # Vector chart: two axes + a diagonal data line + a bezier curve.
    c.setStrokeColor(rl_colors.black)
    c.setLineWidth(1)
    c.line(100, 200, 100, 400)       # Y axis
    c.line(100, 200, 400, 200)       # X axis
    c.line(100, 200, 400, 400)       # data line (diagonal — no grid)
    c.bezier(100, 300, 200, 380, 300, 320, 400, 400)   # trend curve
    # Raster image in a different region of the page.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), 100, 450, width=200, height=100)
    c.showPage()
    c.save()


def _pdf_with_table_only(path) -> None:
    """One page: a proper grid table (4 rows × 5 cols) — no other vector ink.
    pdfplumber's find_tables() should claim the cluster so no figure bbox
    survives table exclusion."""
    from reportlab.lib import colors as rl_colors
    c = canvas.Canvas(str(path), pagesize=letter)
    # Add text so content_type is 'text'.
    t = c.beginText(72, 750)
    for _ in range(4):
        t.textLine("Text to clear the sparse threshold.")
    c.drawText(t)
    c.setStrokeColor(rl_colors.black)
    c.setLineWidth(1)
    # 4 horizontal + 5 vertical lines → a 3×4 grid pdfplumber recognises.
    for row in range(4):
        y = 600 - row * 30
        c.line(100, y, 400, y)
    for col in range(5):
        x = 100 + col * 75
        c.line(x, 600, x, 510)
    c.showPage()
    c.save()


def test_vector_figure_and_raster_both_captured(tmp_path):
    """(a) A page with a vector chart AND a raster image must produce BOTH a
    vector-region crop (image_index_on_page >= 1000) and a raster crop
    (image_index_on_page < 1000). The two run concurrently, not gated on each
    other."""
    src = tmp_path / "vector_and_raster.pdf"
    _pdf_with_vector_figure_and_raster(src, _png_image())
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

    assert result.page_count == 1
    page = result.pages[0]
    assert page.content_type == "text"

    indices = [e.image_index_on_page for e in page.embedded_images]
    # At least one raster-embed crop (index 1-based from page.images).
    raster_crops = [i for i in indices if i < 1000]
    assert raster_crops, (
        f"Expected at least one raster-embed crop (index < 1000), got indices {indices}"
    )
    # At least one vector-figure crop (index >= 1000).
    vector_crops = [i for i in indices if i >= 1000]
    assert vector_crops, (
        f"Expected at least one vector-figure crop (index >= 1000), got indices {indices}"
    )
    # Every crop is a valid PNG that Pillow can open.
    for emb in page.embedded_images:
        decoded = Image.open(io.BytesIO(emb.png_bytes))
        assert decoded.size[0] > 0 and decoded.size[1] > 0


def test_table_only_page_produces_no_vector_crop(tmp_path):
    """(b) A page whose only vector ink forms a proper table grid must produce
    NO vector-figure crop — the table exclusion pass strips the cluster."""
    src = tmp_path / "table_only.pdf"
    _pdf_with_table_only(src)
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

    assert result.page_count == 1
    page = result.pages[0]
    vector_crops = [
        e for e in page.embedded_images if e.image_index_on_page >= 1000
    ]
    assert vector_crops == [], (
        f"Table-only page should have no vector crops, got {len(vector_crops)}"
    )


def test_text_only_page_produces_no_vector_crop(tmp_path):
    """(c) A page with only text (no vector primitives) must produce no
    vector-figure crop. The ink threshold short-circuits the shapely pass."""
    src = tmp_path / "text_only.pdf"
    _text_pdf(src, ["A page of plain text with no charts or figures. " * 6])
    result = pp.convert(src, sparse_text_threshold=100, ocr_min_confidence=30)

    assert result.page_count == 1
    page = result.pages[0]
    assert page.embedded_images == [], (
        f"Text-only page should have no embedded images, got {page.embedded_images}"
    )
