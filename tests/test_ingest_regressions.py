"""Regression tests for real past bugs in ingestion (scribe, parsers, OCR). Each
section names the issue it guards.
"""

from __future__ import annotations

import io

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import pytest

from bartleby.commands import scribe
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.ingest import classify
from bartleby.ingest import ocr
from bartleby.ingest import parsers
from bartleby.ingest import pdfplumber as pp
from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS
from bartleby.providers.base import VlmDescription
import bartleby.config
import bartleby.project


# ---------- scribe: #225/#313 dedup, #164 caption resume + retry cap,
# #307 parse retry cap, #227 worker output, #235 HTML-as-PDF ----------


def _emb(seed: float, n: int) -> list[list[float]]:
    return [
        [seed + 0.0001 * i for _ in range(EMBEDDING_DIM)] for i in range(n)
    ]


def _parse_config(
    archive_root, *, vision_enabled=False, vision_min_dimension=32, verbose=False,
):
    """A ParseConfig with the per-format helpers' run-wide defaults — the scalar
    block these tests used to thread by hand. Only the fields that actually vary
    across call sites are overridable."""
    return parsers.ParseConfig(
        pdf_converter="pdfplumber", html_converter="docling",
        sparse_text_threshold=100, ocr_min_confidence=30,
        vision_enabled=vision_enabled, vision_max_dimension=1024,
        vision_min_dimension=vision_min_dimension,
        vector_ink_threshold=0, archive_root=archive_root,
        verbose=verbose,
    )


@pytest.fixture
def isolated_project(monkeypatch):
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    projects = bartleby.config.projects_dir()
    projects.mkdir(parents=True, exist_ok=True)

    # Pin ingest to the inline parse path. These end-to-end tests mock embedder /
    # converters / providers, and those monkeypatches don't cross into spawned
    # parse-pool workers — so force max_workers=1 (parse in-process) at the
    # resolver, which holds even for tests that swap in their own load_config.
    monkeypatch.setattr(
        "bartleby.ingest.resolve._resolve_max_workers", lambda *a, **k: 1,
    )

    bartleby.project.create_project("test_proj")
    yield projects


@pytest.fixture
def mock_embed(monkeypatch):
    """Replace embed_texts with a deterministic stub returning correctly-sized vectors."""
    def fake(texts):
        return _emb(0.0, len(texts))
    # Patch every import site (commands.scribe imports it directly).
    monkeypatch.setattr("bartleby.ingest.embed.embed_texts", fake)
    monkeypatch.setattr("bartleby.ingest.images.embed_texts", fake)
    return fake


def _png_bytes(width=100, height=100, color=(20, 200, 50)) -> bytes:
    im = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_with_image(path, image_bytes, *, text="Plenty of text on this page so it is not sparse. " * 3):
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 12)
    t = c.beginText(72, 720)
    for line in text.splitlines() or [text]:
        t.textLine(line)
    c.drawText(t)
    c.drawImage(ImageReader(io.BytesIO(image_bytes)), 100, 400, width=200, height=100)
    c.showPage()
    c.save()


def _text_pdf(path, text="Plenty of text on this page so it is not sparse. " * 5):
    """A real, image-free PDF — ingests via pdfplumber without a vision provider."""
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 12)
    t = c.beginText(72, 720)
    for line in (text.splitlines() or [text]):
        t.textLine(line)
    c.drawText(t)
    c.showPage()
    c.save()
    return path


def _write_txt(path, content):
    path.write_text(content, encoding="utf-8")
    return path


class _FlakyVisionProvider:
    """Vision provider whose first ``fail_times`` analyze_image calls raise.

    ``calls`` accumulates across runs when the same instance is returned by a
    patched ``get_provider`` — so a test can assert a caption recovered on a
    later run, or that a capped unit stops reaching the VLM at all.
    """
    name = "flaky-vision"

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def summarize(self, *a, **k):  # protocol completeness
        raise NotImplementedError

    def analyze_image(self, image_bytes, *, model, temperature, media_type="image/jpeg"):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("VLM unavailable")
        return VlmDescription(description="A recovered image.", notes="")


def _vision_pdf_config():
    return {
        "summary_depth": "none",
        "pdf_converter": "pdfplumber", "html_converter": "docling",
        "sparse_text_threshold": 100, "ocr_min_confidence": 30,
        "vision_provider": "stub", "vision_model": "stub-vl:1",
        "vision_max_dimension": 1024, "vision_min_dimension": 32,
    }


def _text_pdf_config():
    """Image-free pdfplumber ingest — no vision provider, no summary pass, so the
    only stage that can fail (or be capped) is parse."""
    return {
        "summary_depth": "none",
        "pdf_converter": "pdfplumber", "html_converter": "docling",
        "sparse_text_threshold": 100, "ocr_min_confidence": 30,
    }


def test_scribe_skips_within_run_duplicate(isolated_project, tmp_path, mock_embed):
    """Two byte-identical files (different names) in ONE run persist a single
    document instead of crashing on documents.file_hash UNIQUE (#225). The DB
    lookup can't catch the in-run twin (neither is committed yet) — _classify's
    queued-hash dedup does."""
    a = _write_txt(tmp_path / "a.txt", "Same content")
    b = _write_txt(tmp_path / "b.txt", "Same content")
    scribe.main(project="test_proj", files=[str(a), str(b)])

    conn = open_db("test_proj")
    try:
        n = conn.cursor().execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert n == 1
    finally:
        conn.close()


def test_persist_parse_reuses_existing_file_hash(
    isolated_project, tmp_path, mock_embed
):
    """persist_parse on a file_hash already in documents returns the existing id
    rather than tripping the UNIQUE constraint — the write-site guard behind
    _classify's dedup (#225)."""
    txt = _write_txt(tmp_path / "doc.txt", "A document body with real words to chunk.")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe.Writer(conn)
        parsed = parsers._parse_document(
            txt, ".txt", _parse_config(archive_root),
            file_hash="dup", file_name="doc.txt",
        )
        first = writer.persist_parse(parsed)
        second = writer.persist_parse(parsed)
        assert second == first
        n = conn.cursor().execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert n == 1
    finally:
        conn.close()


def test_parse_document_rejects_html_saved_as_pdf(tmp_path):
    """A `.pdf` that is really an HTML error page is rejected at dispatch with a
    clear reason, before either PDF backend touches it (#235). The error rides
    the existing parse-failure path into failed_ingests via _parse_request."""
    src = tmp_path / "ViewDoc.pdf"
    src.write_bytes(b"\r\n\r\n<!DOCTYPE html><html><body>portal error</body></html>")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    with pytest.raises(pp.NotAPdfError) as exc:
        parsers._parse_document(
            src, ".pdf", _parse_config(archive_root),
            file_hash="h", file_name="ViewDoc.pdf",
        )
    assert "HTML page" in str(exc.value)
    assert "No /Root object" not in str(exc.value)


def test_parse_image_routes_routes_sub_minimum_warning_off_the_console(
    tmp_path, monkeypatch
):
    """The observed #227 corruptor: a sub-minimum image notice must go to
    ``on_warn`` (routed to the parent), never the console — this code runs in a
    spawn worker with no Live display, so a console write would stomp the bar."""
    # Force every prepared image below the VLM minimum so the skip branch fires.
    monkeypatch.setattr(
        parsers.image_pipeline, "is_below_vlm_minimum", lambda *a, **k: True
    )
    # A worker must never touch the console; fail loudly if it tries.
    monkeypatch.setattr(
        scribe.console, "warn",
        lambda *a, **k: pytest.fail("worker-side parse called console.warn"),
    )

    warnings: list[str] = []
    route = parsers._ImageRoute(
        bytes_=_png_bytes(), page_number=3, image_index_on_page=0,
    )
    images = parsers._parse_image_routes(
        [route],
        _parse_config(tmp_path / "archive", vision_enabled=True, vision_min_dimension=128),
        on_warn=warnings.append,
    )

    assert images == []
    assert len(warnings) == 1
    assert "below the 128px vision minimum" in warnings[0]


def test_classify_dedupes_byte_identical_incomplete_resume(isolated_project, tmp_path):
    """Two byte-identical copies of an incomplete, already-parsed file resume the
    one document once: the first lands in to_resume, the twin is diverted to
    duplicates. Without the dedup both resolve to the same document_id and both
    land in to_resume, so the incomplete tally double-counts a phantom unit (#313)."""
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"identical image bytes")
    b.write_bytes(b"identical image bytes")
    file_hash = classify._hash_file(a)
    assert classify._hash_file(b) == file_hash

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_hash, "a.jpg", str(a), None, 0),
        )
        doc_id = conn.last_insert_rowid()
        # analysis_json IS NULL → uncaptioned → the document is incomplete, so the
        # first copy resumes rather than skips.
        cur.execute(
            "INSERT INTO images (file_hash, file_path, width, height, "
            "analysis_json, analysis_model) VALUES (?, ?, ?, ?, NULL, NULL)",
            ("imgblob", str(a), 100, 100),
        )
        image_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_images "
            "(document_id, image_id, page_number, image_index_on_page) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, image_id, None, 0),
        )
        writer = scribe.Writer(conn)
        to_parse, to_resume, skipped, duplicates = classify._classify(
            writer, [(a, ".jpg"), (b, ".jpg")], vision_enabled=True,
        )
        assert to_parse == []
        assert [r.document_id for r in to_resume] == [doc_id]
        assert duplicates == ["b.jpg"]
        assert skipped == []
    finally:
        conn.close()


def test_scribe_resumes_missing_caption_without_reparsing(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """The caption-loss bug fix: when the VLM dies after the text chunks land,
    the parse stays durable; a later run captions only the missing image and
    never re-parses or re-embeds the document body."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config", _vision_pdf_config,
    )
    vision = _FlakyVisionProvider(fail_times=1)
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: vision,
    )

    # Count real pdfplumber parses to prove the second run doesn't re-parse.
    real_convert = pp.convert
    parses = {"n": 0}

    def counting_convert(*a, **k):
        parses["n"] += 1
        return real_convert(*a, **k)

    monkeypatch.setattr(
        "bartleby.ingest.parsers.pdfplumber_pipeline.convert", counting_convert,
    )

    pdf = tmp_path / "doc.pdf"
    _pdf_with_image(pdf, _png_bytes(), text="Durable body text " * 12)

    # Run 1: the VLM fails. Text chunks land; the image is recorded uncaptioned.
    # The uncaptioned image is an unresolved unit, so the run exits non-zero (#311).
    with pytest.raises(SystemExit) as exc:
        scribe.main(project="test_proj", files=str(pdf))
    assert exc.value.code == 1
    assert parses["n"] == 1
    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        doc_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0]
        assert doc_chunks >= 1                               # parse durable
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert cur.execute(
            "SELECT analysis_json FROM images"
        ).fetchone()[0] is None                              # uncaptioned
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='image'"
        ).fetchone()[0] == 0
        stage, attempts = cur.execute(
            "SELECT stage, attempts FROM failed_ingests"
        ).fetchone()
        assert stage == "caption" and attempts == 1
    finally:
        conn.close()

    # Run 2: the VLM recovers. The image is captioned; nothing is re-parsed.
    scribe.main(project="test_proj", files=str(pdf))
    assert parses["n"] == 1                                  # NOT re-parsed
    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0] == doc_chunks                        # body untouched
        assert cur.execute(
            "SELECT analysis_json FROM images"
        ).fetchone()[0] is not None                          # captioned
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='image'"
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM failed_ingests"
        ).fetchone()[0] == 0                                 # failure cleared
    finally:
        conn.close()

    # One failed call, one successful — the missing image, captioned once.
    assert vision.calls == 2


def test_scribe_caps_caption_retries_and_stops_calling_vlm(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """A deterministically-failing caption is retried up to the cap, recorded,
    then never sent to the VLM again — so a poison image can't loop forever and
    can't silently read as done."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config", _vision_pdf_config,
    )
    vision = _FlakyVisionProvider(fail_times=10_000)  # always fails
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: vision,
    )

    pdf = tmp_path / "doc.pdf"
    _pdf_with_image(pdf, _png_bytes(), text="Body text " * 12)

    # Each run makes exactly one more attempt until the cap is reached. The
    # image never captions, so every run exits non-zero on the unresolved unit (#311).
    for _ in range(MAX_INGEST_ATTEMPTS):
        with pytest.raises(SystemExit) as exc:
            scribe.main(project="test_proj", files=str(pdf))
        assert exc.value.code == 1
    assert vision.calls == MAX_INGEST_ATTEMPTS

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        stage, attempts = cur.execute(
            "SELECT stage, attempts FROM failed_ingests"
        ).fetchone()
        assert stage == "caption" and attempts == MAX_INGEST_ATTEMPTS
        assert cur.execute(
            "SELECT analysis_json FROM images"
        ).fetchone()[0] is None                              # still uncaptioned
    finally:
        conn.close()

    # A further run is a no-op against the VLM — the unit is capped. It's still
    # unresolved, so the run keeps exiting non-zero rather than reading green (#311).
    with pytest.raises(SystemExit) as exc:
        scribe.main(project="test_proj", files=str(pdf))
    assert exc.value.code == 1
    assert vision.calls == MAX_INGEST_ATTEMPTS
    conn = open_db("test_proj")
    try:
        assert conn.cursor().execute(
            "SELECT attempts FROM failed_ingests"
        ).fetchone()[0] == MAX_INGEST_ATTEMPTS               # not bumped past cap
    finally:
        conn.close()


def test_scribe_caps_parse_retries_and_stops_reparsing(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """A deterministically-failing parse is retried up to the cap, recorded, then
    never re-parsed again — the parse stage now honours MAX_INGEST_ATTEMPTS like
    caption/summary, so a corrupt file can't loop the most expensive stage
    forever and can't silently read as done (#307)."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config", _text_pdf_config,
    )

    parses = {"n": 0}

    def failing_convert(*a, **k):
        parses["n"] += 1
        raise RuntimeError("corrupt PDF")

    monkeypatch.setattr(
        "bartleby.ingest.parsers.pdfplumber_pipeline.convert", failing_convert,
    )

    pdf = _text_pdf(tmp_path / "doc.pdf", text="Body text " * 12)

    # Each run makes exactly one more parse attempt until the cap is reached.
    # The parse never lands, so every run exits non-zero on the unresolved unit (#311).
    for _ in range(MAX_INGEST_ATTEMPTS):
        with pytest.raises(SystemExit) as exc:
            scribe.main(project="test_proj", files=str(pdf))
        assert exc.value.code == 1
    assert parses["n"] == MAX_INGEST_ATTEMPTS

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        stage, attempts = cur.execute(
            "SELECT stage, attempts FROM failed_ingests"
        ).fetchone()
        assert stage == "parse" and attempts == MAX_INGEST_ATTEMPTS
        # The parse never succeeded — nothing persisted.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    finally:
        conn.close()

    # A further run is a no-op against the converter — the unit is capped. Still
    # unresolved, so it keeps exiting non-zero rather than reading green (#311).
    with pytest.raises(SystemExit) as exc:
        scribe.main(project="test_proj", files=str(pdf))
    assert exc.value.code == 1
    assert parses["n"] == MAX_INGEST_ATTEMPTS                # NOT re-parsed
    conn = open_db("test_proj")
    try:
        assert conn.cursor().execute(
            "SELECT attempts FROM failed_ingests"
        ).fetchone()[0] == MAX_INGEST_ATTEMPTS               # not bumped past cap
    finally:
        conn.close()


# ---------- pdfplumber: #309 OCR→VLM fallback ----------


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


# ---------- ocr: #309 legible error when tesseract is missing ----------


def _blank_image() -> bytes:
    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
