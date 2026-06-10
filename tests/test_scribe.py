"""End-to-end ingest test for `.txt` with mocked embedder and provider."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import bartleby.config
import bartleby.db.connection
import bartleby.project
from bartleby.commands import scribe
from bartleby.ingest import parsers
from bartleby.ingest import classify
from bartleby.ingest import caption
from bartleby.ingest import parse
from bartleby.ingest import summary
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.ingest.chunk import resolve_extension
from bartleby.providers.base import DocumentSummary, VlmDescription


def _emb(seed: float, n: int) -> list[list[float]]:
    return [
        [seed + 0.0001 * i for _ in range(EMBEDDING_DIM)] for i in range(n)
    ]


class _StubProvider:
    name = "stub"

    def __init__(
        self,
        text: str = "## Summary\n\nA stub summary.",
        title: str = "Stub Title",
        description: str = "Stub one-line description.",
        authored_date: str | None = None,
    ):
        self.text = text
        self.title = title
        self.description = description
        self.authored_date = authored_date
        self.calls = 0
        self.last_document_text: str | None = None

    def summarize(self, document_text, *, model, temperature,
                  reasoning_effort=None):
        self.calls += 1
        self.last_document_text = document_text
        return DocumentSummary(
            title=self.title, description=self.description, text=self.text,
            authored_date=self.authored_date,
        )


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    config_path = tmp_path / "config.yaml"

    monkeypatch.setattr(bartleby.config, "BARTLEBY_DIR", tmp_path)
    monkeypatch.setattr(bartleby.config, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(bartleby.project, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.db.connection, "PROJECTS_DIR", projects)

    # Pin ingest to the inline parse path. These end-to-end tests mock embedder /
    # converters / providers, and those monkeypatches don't cross into spawned
    # parse-pool workers — so force max_workers=1 (parse in-process) at the
    # resolver, which holds even for tests that swap in their own load_config.
    # The pool is exercised separately in test_ingest_pool.py with a picklable,
    # model-free parse_fn.
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


class _StubVisionProvider:
    name = "stub-vision"

    def __init__(self, description: VlmDescription | None = None):
        self.description = description or VlmDescription(
            description="A test image.", notes="",
        )
        self.calls = 0

    def summarize(self, *a, **k):  # protocol completeness
        raise NotImplementedError

    def analyze_image(self, image_bytes, *, model, media_type="image/jpeg"):
        self.calls += 1
        return self.description


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


def test_scribe_ingests_txt_without_llm(isolated_project, tmp_path, mock_embed):
    src = _write_txt(tmp_path / "doc.txt",
                     "Hello world. This is a small text document for ingestion.")
    scribe.main(project="test_proj", files=str(src), verbose=False)

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        docs = cur.execute(
            "SELECT file_name, page_count, token_count FROM documents"
        ).fetchall()
        assert len(docs) == 1
        assert docs[0][0] == "doc.txt"
        assert docs[0][1] is None              # txt has no page count
        assert docs[0][2] > 0                  # tiktoken counted real tokens

        n_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0]
        assert n_chunks >= 1

        assert cur.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == n_chunks
        assert cur.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == n_chunks

        # No summary written because no provider configured.
        assert cur.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0
    finally:
        conn.close()


def test_scribe_dedupes_identical_files(isolated_project, tmp_path, mock_embed):
    src = _write_txt(tmp_path / "doc.txt", "Same content")
    scribe.main(project="test_proj", files=str(src))
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        n = conn.cursor().execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert n == 1
    finally:
        conn.close()


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
    from bartleby.commands import scribe as scribe_module

    txt = _write_txt(tmp_path / "doc.txt", "A document body with real words to chunk.")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe_module.Writer(conn)
        parsed = parsers._parse_document(
            txt, ".txt", file_hash="dup", file_name="doc.txt",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
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
    from bartleby.commands import scribe as scribe_module
    from bartleby.ingest.pdfplumber import NotAPdfError

    src = tmp_path / "ViewDoc.pdf"
    src.write_bytes(b"\r\n\r\n<!DOCTYPE html><html><body>portal error</body></html>")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    with pytest.raises(NotAPdfError) as exc:
        parsers._parse_document(
            src, ".pdf", file_hash="h", file_name="ViewDoc.pdf",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
        )
    assert "HTML page" in str(exc.value)
    assert "No /Root object" not in str(exc.value)


def test_scribe_writes_summary_when_provider_configured(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    # Configure summarization in the loaded config.
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic",
            "model": "test-model",
            "temperature": 0.0,
            "max_summarize_tokens": 50_000,
        },
    )

    stub = _StubProvider(text="## Stub summary\n\nKey points appear here.")
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: stub,
    )
    # Bypass docling for the summary chunking too — return one chunk.
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    src = _write_txt(tmp_path / "doc.txt", "Hello world summarized.")
    scribe.main(project="test_proj", files=str(src))

    assert stub.calls == 1

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        summaries = cur.execute(
            "SELECT title, description, text, model FROM summaries"
        ).fetchall()
        assert len(summaries) == 1
        title, description, text, model = summaries[0]
        assert title == "Stub Title"
        assert description == "Stub one-line description."
        assert text.startswith("## Stub summary")
        assert model == "test-model"

        sum_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='summary'"
        ).fetchone()[0]
        assert sum_chunks == 1
    finally:
        conn.close()


def test_scribe_summarizes_existing_document_on_later_run(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """Summarization is its own pass over whatever the DB still lacks (issue
    #167): a document parsed on an earlier run with summaries off is summarized
    on a later run with summaries on — without being re-parsed. The pass picks
    it up from the DB even though classification skips it as already ingested."""
    src = _write_txt(tmp_path / "doc.txt", "Hello world summarized.")

    # First run: no summary provider configured → parse only, no summary.
    scribe.main(project="test_proj", files=str(src))
    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0
    finally:
        conn.close()

    # Second run: summaries now configured. The file is already parsed (so it's
    # classified as skipped), yet the summarize pass still summarizes it.
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot", "provider": "anthropic",
            "model": "test-model", "temperature": 0.0,
            "max_summarize_tokens": 50_000,
        },
    )
    stub = _StubProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **kwargs: stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    scribe.main(project="test_proj", files=str(src))
    assert stub.calls == 1  # summarized exactly once

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # Not re-parsed: still a single document row, now with one summary.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 1
    finally:
        conn.close()


def test_scribe_ingests_pdf_via_pdfplumber_with_embedded_image(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "pdf_converter": "pdfplumber",
            "html_converter": "docling",
            "sparse_text_threshold": 100,
            "ocr_min_confidence": 30,
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: vision,
    )

    pdf = tmp_path / "doc.pdf"
    _pdf_with_image(pdf, _png_bytes())

    scribe.main(project="test_proj", files=str(pdf))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # One document row, with chunks tagged content_type='text'.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        n_doc_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0]
        assert n_doc_chunks >= 1
        # Embedded image processed: rows in images + document_images, plus
        # one image chunk per non-empty analysis field.
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] >= 1
        assert cur.execute("SELECT COUNT(*) FROM document_images").fetchone()[0] >= 1
        n_image_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='image'"
        ).fetchone()[0]
        # Binary classification: a tiny color-block image has no Tesseract text
        # → kind='scene' → one image_description chunk per image.
        assert n_image_chunks == 1
        # Vision provider called at least once for the embedded image.
        assert vision.calls >= 1
    finally:
        conn.close()


def test_scribe_ingests_pdf_via_docling_without_second_parse(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """Docling-backend images come from the docling pass, not a pdfplumber re-parse."""
    from bartleby.ingest.docling import DoclingChunk, DoclingImage, DoclingResult

    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "pdf_converter": "docling",
            "html_converter": "docling",
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: vision,
    )

    # Stub the docling pass: one text chunk + one embedded picture. Capturing
    # the call lets us assert the image side-pass was requested in the same pass.
    calls = {}

    def fake_convert(path, *, extract_images=False):
        calls["extract_images"] = extract_images
        return DoclingResult(
            full_text="Body text from docling.",
            page_count=1,
            chunks=[DoclingChunk(
                text="Body text from docling.",
                section_heading=None, content_type="text", page_number=1,
            )],
            images=[DoclingImage(
                png_bytes=_png_bytes(), page_number=1, image_index_on_page=1,
            )] if extract_images else [],
        )

    monkeypatch.setattr("bartleby.ingest.docling.convert", fake_convert)
    # The whole point of #2: the docling path must not re-parse via pdfplumber.
    def _boom(*a, **k):
        raise AssertionError("docling path must not call pdfplumber")
    monkeypatch.setattr(
        "bartleby.ingest.parsers.pdfplumber_pipeline.convert", _boom
    )

    pdf = tmp_path / "doc.pdf"
    _pdf_with_image(pdf, _png_bytes())

    scribe.main(project="test_proj", files=str(pdf))

    assert calls["extract_images"] is True
    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM document_images"
        ).fetchone()[0] == 1
        assert vision.calls >= 1
    finally:
        conn.close()


def test_scribe_dedupes_identical_images_across_documents(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "pdf_converter": "pdfplumber",
            "html_converter": "docling",
            "sparse_text_threshold": 100,
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: vision,
    )

    # Two distinct PDFs, both containing the same image (same byte content).
    image_bytes = _png_bytes(width=120, height=80, color=(33, 99, 200))
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _pdf_with_image(pdf_a, image_bytes, text="Doc A text " * 12)
    _pdf_with_image(pdf_b, image_bytes, text="Doc B text " * 12)

    scribe.main(project="test_proj", files=str(pdf_a))
    calls_after_first = vision.calls
    scribe.main(project="test_proj", files=str(pdf_b))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # Two documents, ONE shared image row, TWO join rows.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM document_images").fetchone()[0] == 2
    finally:
        conn.close()

    # Second pass made no additional VLM calls — the dedupe is real.
    assert vision.calls == calls_after_first


def test_scribe_captions_many_images_concurrently_in_one_run(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """#166: captioning is its own concurrent phase. Several documents parsed in
    one run all get every image captioned through the caption thread pool — not
    just the first — with the DB write staying on the single Writer thread."""
    config = _vision_pdf_config()
    config["caption_workers"] = 3
    monkeypatch.setattr("bartleby.commands.scribe.load_config", lambda: config)
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: vision,
    )

    pdfs = []
    for i in range(4):
        pdf = tmp_path / f"doc{i}.pdf"
        # A distinct colour per doc → distinct byte-hash, so no cross-doc dedup.
        _pdf_with_image(
            pdf, _png_bytes(color=(10 + i * 30, 80, 200 - i * 20)),
            text=f"Body of document {i} " * 12,
        )
        pdfs.append(str(pdf))

    scribe.main(project="test_proj", files=pdfs)

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 4
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 4
        # The authoritative check (race-free, written on the writer thread):
        # every image was captioned — none left with a NULL analysis.
        assert cur.execute(
            "SELECT COUNT(*) FROM images WHERE analysis_json IS NULL"
        ).fetchone()[0] == 0
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='image'"
        ).fetchone()[0] == 4
    finally:
        conn.close()


def test_scribe_dedupes_shared_image_within_one_run(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """Two documents sharing an image, ingested in the same run, caption it once:
    the caption phase keys pending work by image row, so a shared row is analyzed
    a single time even though it joins two documents."""
    config = _vision_pdf_config()
    config["caption_workers"] = 1   # single-threaded → exact VLM call count
    monkeypatch.setattr("bartleby.commands.scribe.load_config", lambda: config)
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: vision,
    )

    image_bytes = _png_bytes(width=120, height=80, color=(33, 99, 200))
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _pdf_with_image(pdf_a, image_bytes, text="Doc A text " * 12)
    _pdf_with_image(pdf_b, image_bytes, text="Doc B text " * 12)

    scribe.main(project="test_proj", files=[str(pdf_a), str(pdf_b)])

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM document_images").fetchone()[0] == 2
        assert cur.execute(
            "SELECT analysis_json FROM images"
        ).fetchone()[0] is not None
    finally:
        conn.close()

    # Analyzed exactly once despite living in two documents.
    assert vision.calls == 1


def test_scribe_one_caption_failure_does_not_block_the_rest(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """A single image whose VLM call raises is recorded as a caption failure;
    the other images in the same concurrent run are still captioned — one bad
    image never tears down the phase."""
    import threading

    class _OneFailVision:
        name = "one-fail-vision"

        def __init__(self):
            self.calls = 0
            self._lock = threading.Lock()

        def summarize(self, *a, **k):  # protocol completeness
            raise NotImplementedError

        def analyze_image(self, image_bytes, *, model, media_type="image/jpeg"):
            # Lock so the "first call fails" rule is deterministic under the pool.
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                raise RuntimeError("VLM unavailable")
            return VlmDescription(description="ok", notes="")

    config = _vision_pdf_config()
    config["caption_workers"] = 3
    monkeypatch.setattr("bartleby.commands.scribe.load_config", lambda: config)
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: _OneFailVision(),
    )

    pdfs = []
    for i in range(3):
        pdf = tmp_path / f"d{i}.pdf"
        # Long enough not to trip the sparse-page render (one image per doc).
        _pdf_with_image(
            pdf, _png_bytes(color=(20 + i * 40, 90, 150)),
            text=f"Body of document {i} with plenty of words so it is not sparse. " * 3,
        )
        pdfs.append(str(pdf))

    scribe.main(project="test_proj", files=pdfs)

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 3
        # Exactly one image left uncaptioned (the failed call); the other two land.
        assert cur.execute(
            "SELECT COUNT(*) FROM images WHERE analysis_json IS NULL"
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM images WHERE analysis_json IS NOT NULL"
        ).fetchone()[0] == 2
        assert cur.execute(
            "SELECT stage FROM failed_ingests"
        ).fetchone()[0] == "caption"
    finally:
        conn.close()


def test_resolve_caption_workers_defaults_and_timings():
    from bartleby.ingest import resolve as scribe_module
    from bartleby.lib.consts import DEFAULT_CAPTION_WORKERS

    # Default when unset or zero; explicit value respected.
    assert scribe_module._resolve_caption_workers(
        {}, timings=False) == DEFAULT_CAPTION_WORKERS
    assert scribe_module._resolve_caption_workers(
        {"caption_workers": 0}, timings=False) == DEFAULT_CAPTION_WORKERS
    assert scribe_module._resolve_caption_workers(
        {"caption_workers": 9}, timings=False) == 9
    # A cloud vision provider keeps the configured/default count.
    assert scribe_module._resolve_caption_workers(
        {"vision_provider": "anthropic"}, timings=False) == DEFAULT_CAPTION_WORKERS
    # --timings forces a sequential baseline regardless of config.
    assert scribe_module._resolve_caption_workers(
        {"caption_workers": 9}, timings=True) == 1


def test_resolve_caption_workers_clamps_ollama(monkeypatch):
    from bartleby.ingest import resolve as scribe_module

    warnings: list[str] = []
    monkeypatch.setattr(scribe_module.console, "warn", warnings.append)

    # Ollama serializes (OLLAMA_NUM_PARALLEL=1): clamp to 1, silent at default.
    assert scribe_module._resolve_caption_workers(
        {"vision_provider": "ollama"}, timings=False) == 1
    assert warnings == []
    # An explicit count > 1 is ignored (still 1) and warns.
    assert scribe_module._resolve_caption_workers(
        {"vision_provider": "ollama", "caption_workers": 8}, timings=False) == 1
    assert any("caption_workers > 1 ignored" in w for w in warnings)


def _summary_config(**overrides):
    """A summaries-on config for the LLM-backed ingest tests."""
    return {
        "summary_depth": "one-shot",
        "provider": "anthropic", "model": "m",
        "temperature": 0.0, "max_summarize_tokens": 50_000,
        **overrides,
    }


def test_scribe_summarizes_many_documents_concurrently_in_one_run(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """#188: summarization is its own concurrent pass. Several documents parsed in
    one run all get summarized through the summarize thread pool — not just the
    first — with the DB write staying on the single Writer thread."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: _summary_config(summarize_workers=3),
    )
    stub = _StubProvider(text="summary body")
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    srcs = [
        str(_write_txt(tmp_path / f"doc{i}.txt", f"Body of document {i}."))
        for i in range(4)
    ]
    scribe.main(project="test_proj", files=srcs)

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 4
        # Every document summarized — race-free DB check on the writer thread.
        assert cur.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 4
    finally:
        conn.close()


def test_scribe_one_summary_failure_does_not_block_the_rest(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """A single document whose summarize call raises is recorded as a summary
    failure; the others in the same concurrent run still summarize — one bad
    document never tears down the pass."""
    import threading

    class _OneFailLLM:
        name = "one-fail-llm"

        def __init__(self):
            self.calls = 0
            self._lock = threading.Lock()

        def analyze_image(self, *a, **k):  # protocol completeness
            raise NotImplementedError

        def summarize(self, document_text, *, model, temperature,
                      reasoning_effort=None):
            # Lock so the "first call fails" rule is deterministic under the pool.
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                raise RuntimeError("LLM unavailable")
            return DocumentSummary(
                title="t", description="d", text="ok", authored_date=None,
            )

    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: _summary_config(summarize_workers=3),
    )
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: _OneFailLLM(),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    srcs = [
        str(_write_txt(tmp_path / f"d{i}.txt", f"Body of document {i}."))
        for i in range(3)
    ]
    scribe.main(project="test_proj", files=srcs)

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # Exactly one document left without a summary (the failed call); two land.
        assert cur.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 2
        assert cur.execute(
            "SELECT stage FROM failed_ingests"
        ).fetchone()[0] == "summary"
    finally:
        conn.close()


def test_resolve_summarize_workers_defaults_and_timings():
    from bartleby.ingest import resolve as scribe_module
    from bartleby.lib.consts import DEFAULT_SUMMARIZE_WORKERS

    # Default when unset or zero; explicit value respected.
    assert scribe_module._resolve_summarize_workers(
        {}, timings=False) == DEFAULT_SUMMARIZE_WORKERS
    assert scribe_module._resolve_summarize_workers(
        {"summarize_workers": 0}, timings=False) == DEFAULT_SUMMARIZE_WORKERS
    assert scribe_module._resolve_summarize_workers(
        {"summarize_workers": 9}, timings=False) == 9
    # A cloud LLM provider keeps the configured/default count.
    assert scribe_module._resolve_summarize_workers(
        {"provider": "openai"}, timings=False) == DEFAULT_SUMMARIZE_WORKERS
    # --timings forces a sequential baseline regardless of config.
    assert scribe_module._resolve_summarize_workers(
        {"summarize_workers": 9}, timings=True) == 1


def test_resolve_summarize_workers_clamps_ollama(monkeypatch):
    from bartleby.ingest import resolve as scribe_module

    warnings: list[str] = []
    monkeypatch.setattr(scribe_module.console, "warn", warnings.append)

    # Ollama serializes (OLLAMA_NUM_PARALLEL=1): clamp to 1, silent at default.
    assert scribe_module._resolve_summarize_workers(
        {"provider": "ollama"}, timings=False) == 1
    assert warnings == []
    # An explicit count > 1 is ignored (still 1) and warns.
    assert scribe_module._resolve_summarize_workers(
        {"provider": "ollama", "summarize_workers": 8}, timings=False) == 1
    assert any("summarize_workers > 1 ignored" in w for w in warnings)


def test_summarize_all_progress_callback_fires_per_document(
    isolated_project, tmp_path, mock_embed
):
    """The summarize pass invokes on_progress(0, N) then once per document."""
    from bartleby.commands import scribe as scribe_module

    txt = _write_txt(tmp_path / "doc.txt", "A document body with real words to chunk.")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe_module.Writer(conn)
        parsed = parsers._parse_document(
            txt, ".txt", file_hash="h", file_name="doc.txt",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
        )
        writer.persist_parse(parsed)

        pending = writer.documents_needing_summary()
        seen: list[tuple[int, int]] = []
        owed, _times = summary._summarize_all(
            writer, pending,
            llm_provider=_StubProvider(), llm_model="m",
            temperature=0.0, max_summarize_tokens=1000,
            summarize_workers=1, timings=False,
            on_progress=lambda done, total: seen.append((done, total)),
        )
    finally:
        conn.close()

    assert seen, "expected at least one progress callback for one document"
    assert seen[0] == (0, 1)
    assert seen[-1] == (1, 1)
    assert owed == 0


def test_summarize_all_lane_callback_reports_document(
    isolated_project, tmp_path, mock_embed
):
    """on_lane fires per summarized document with its file name and stage."""
    from bartleby.commands import scribe as scribe_module

    txt = _write_txt(tmp_path / "doc.txt", "A document body with real words to chunk.")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe_module.Writer(conn)
        parsed = parsers._parse_document(
            txt, ".txt", file_hash="h", file_name="doc.txt",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
        )
        writer.persist_parse(parsed)

        pending = writer.documents_needing_summary()
        lanes: list[tuple[object, str, str]] = []
        summary._summarize_all(
            writer, pending,
            llm_provider=_StubProvider(), llm_model="m",
            temperature=0.0, max_summarize_tokens=1000,
            summarize_workers=1, timings=False,
            on_progress=None,
            on_lane=lambda key, item, stage: lanes.append((key, item, stage)),
        )
    finally:
        conn.close()

    assert lanes == [(lanes[0][0], "doc.txt", "summarizing")]


def test_summarize_all_skips_capped_document(
    isolated_project, tmp_path, mock_embed
):
    """A document already capped on the summary stage is surfaced as owed and
    never re-handed to the model — counts incomplete, no provider call."""
    from bartleby.commands import scribe as scribe_module
    from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS

    txt = _write_txt(tmp_path / "doc.txt", "A document body with real words to chunk.")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe_module.Writer(conn)
        parsed = parsers._parse_document(
            txt, ".txt", file_hash="caphash", file_name="doc.txt",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
        )
        writer.persist_parse(parsed)
        # Cap the summary stage so the pass must skip rather than retry.
        for _ in range(MAX_INGEST_ATTEMPTS):
            writer.record_failure("caphash", "doc.txt", "summary", RuntimeError("x"))

        stub = _StubProvider()
        owed, _times = summary._summarize_all(
            writer, writer.documents_needing_summary(),
            llm_provider=stub, llm_model="m",
            temperature=0.0, max_summarize_tokens=1000,
            summarize_workers=2, timings=False, on_progress=None,
        )
    finally:
        conn.close()

    assert owed == 1            # surfaced as still-incomplete
    assert stub.calls == 0      # never handed to the model


def test_scribe_ingests_standalone_image_file(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: vision,
    )

    img = tmp_path / "photo.png"
    img.write_bytes(_png_bytes())

    scribe.main(project="test_proj", files=str(img))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        # No document-kind chunks for a standalone image.
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0] == 0
        # But image chunks were written and joined.
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        n_image_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='image'"
        ).fetchone()[0]
        # Binary classification: standalone PNG with no recognizable text →
        # kind='scene' → one image_description chunk.
        assert n_image_chunks == 1
        # document_images row has NULL page_number for standalone files.
        row = cur.execute(
            "SELECT page_number, image_index_on_page FROM document_images"
        ).fetchone()
        assert row == (None, 0)
    finally:
        conn.close()


def test_scribe_skips_sub_minimum_image_without_calling_vlm(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """A thin sub-32px strip is dropped before the VLM (no row, no crash)."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
            "vision_min_dimension": 32,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: vision,
    )

    img = tmp_path / "rule.png"
    img.write_bytes(_png_bytes(width=512, height=24))

    scribe.main(project="test_proj", files=str(img))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # The document row exists, but the sub-minimum image is skipped entirely:
        # no images row, no join, no image chunk.
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM document_images").fetchone()[0] == 0
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='image'"
        ).fetchone()[0] == 0
    finally:
        conn.close()

    # The VLM was never called — no chance to crash the runner.
    assert vision.calls == 0


def test_scribe_skips_image_when_no_vision_provider(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {"summary_depth": "none"},
    )
    img = tmp_path / "photo.png"
    img.write_bytes(_png_bytes())

    scribe.main(project="test_proj", files=str(img))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # No document was created; the file was skipped with a warning.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
    finally:
        conn.close()


def test_scribe_persists_page_number_for_pdfplumber_chunks(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """Document chunks from pdfplumber carry a real page_number column."""
    from bartleby.db.connection import open_db
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "pdf_converter": "pdfplumber",
            "html_converter": "docling",
            "sparse_text_threshold": 100,
            "ocr_min_confidence": 30,
        },
    )

    pdf = tmp_path / "multi.pdf"
    _pdf_with_image(pdf, _png_bytes(), text="Page-one body text " * 10)
    scribe.main(project="test_proj", files=str(pdf))

    conn = open_db("test_proj")
    try:
        rows = conn.cursor().execute(
            "SELECT page_number, section_heading FROM chunks "
            "WHERE source_kind='document'"
        ).fetchall()
        assert rows
        # All pdfplumber-derived document chunks have a real page_number and
        # a NULL section_heading (the old 'page N' hack is gone).
        for page_number, section_heading in rows:
            assert page_number == 1
            assert section_heading is None
    finally:
        conn.close()


def test_parse_document_stage_callback_fires_extract_then_embed(
    isolated_project, tmp_path, mock_embed
):
    """Parse emits its stages in order: extracting → embedding. (Caption fires
    downstream in the main-process drain and summarize in its own later pass —
    not here; parse runs in a pool worker and only carries the parse-side
    stages.)"""
    from bartleby.commands import scribe as scribe_module

    pdf = tmp_path / "doc.pdf"
    _text_pdf(pdf)

    stages: list[str] = []
    parsers._parse_document(
        pdf, ".pdf", file_hash="h", file_name="doc.pdf",
        pdf_converter="pdfplumber", html_converter="docling",
        sparse_text_threshold=100, ocr_min_confidence=30,
        vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
        archive_root=tmp_path / "archive",
        on_stage=stages.append,
    )

    assert stages == ["extracting", "embedding"]


def test_parse_document_reports_page_count(
    isolated_project, tmp_path, mock_embed
):
    """The parse result carries the converter's page count (issue #85)."""
    from bartleby.commands import scribe as scribe_module

    pdf = tmp_path / "doc.pdf"
    _text_pdf(pdf)

    parsed = parsers._parse_document(
        pdf, ".pdf", file_hash="h", file_name="doc.pdf",
        pdf_converter="pdfplumber", html_converter="docling",
        sparse_text_threshold=100, ocr_min_confidence=30,
        vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=32,
        archive_root=tmp_path / "archive",
    )

    assert parsed.page_count == 1


def test_parse_image_routes_routes_sub_minimum_warning_off_the_console(
    tmp_path, monkeypatch
):
    """The observed #227 corruptor: a sub-minimum image notice must go to
    ``on_warn`` (routed to the parent), never the console — this code runs in a
    spawn worker with no Live display, so a console write would stomp the bar."""
    from bartleby.commands import scribe as scribe_module

    # Force every prepared image below the VLM minimum so the skip branch fires.
    monkeypatch.setattr(
        parsers.image_pipeline, "is_below_vlm_minimum", lambda *a, **k: True
    )
    # A worker must never touch the console; fail loudly if it tries.
    monkeypatch.setattr(
        scribe_module.console, "warn",
        lambda *a, **k: pytest.fail("worker-side parse called console.warn"),
    )

    warnings: list[str] = []
    route = parsers._ImageRoute(
        bytes_=_png_bytes(), page_number=3, image_index_on_page=0,
    )
    images = parsers._parse_image_routes(
        [route], archive_root=tmp_path / "archive", vision_enabled=True,
        vision_max_dimension=1024, vision_min_dimension=128,
        on_warn=warnings.append,
    )

    assert images == []
    assert len(warnings) == 1
    assert "page 3" in warnings[0]
    assert "below the 128px vision minimum" in warnings[0]


def test_parse_document_threads_on_warn_to_the_leaf(
    isolated_project, tmp_path, mock_embed
):
    """on_warn reaches the leaf through the full dispatch chain: a standalone
    image parsed with vision off surfaces the 'no vision provider' notice as a
    routed warning, not a console write."""
    from bartleby.commands import scribe as scribe_module

    img = tmp_path / "pic.png"
    img.write_bytes(_png_bytes())

    warnings: list[str] = []
    parsers._parse_document(
        img, ".png", file_hash="h", file_name="pic.png",
        pdf_converter="pdfplumber", html_converter="docling",
        sparse_text_threshold=100, ocr_min_confidence=30,
        vision_enabled=False, vision_max_dimension=1024, vision_min_dimension=128,
        archive_root=tmp_path / "archive",
        on_warn=warnings.append,
    )

    assert warnings == ["Skipping 1 image(s) — no vision provider configured."]


def test_caption_all_progress_callback_fires_per_image(
    isolated_project, tmp_path, mock_embed
):
    """The caption phase invokes on_progress(0, N) then once per image."""
    from bartleby.commands import scribe as scribe_module
    from bartleby.db.connection import open_db

    pdf = tmp_path / "img.pdf"
    _pdf_with_image(pdf, _png_bytes())

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe_module.Writer(conn)
        parsed = parsers._parse_document(
            pdf, ".pdf", file_hash="h", file_name="img.pdf",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=True, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
        )
        document_id = writer.persist_parse(parsed)

        seen: list[tuple[int, int]] = []
        caption._caption_all(
            writer,
            [parse.DocUnit(document_id, "img.pdf", "h")],
            vision_provider=_StubVisionProvider(), vision_model="stub-vl:1",
            vision_enabled=True, caption_workers=1, timings=False,
            on_progress=lambda done, total: seen.append((done, total)),
        )
    finally:
        conn.close()

    assert seen, "expected at least one progress callback for one embedded image"
    assert seen[0] == (0, 1)
    assert seen[-1] == (1, 1)


def test_caption_all_lane_callback_reports_owning_document(
    isolated_project, tmp_path, mock_embed
):
    """on_lane fires per analyzed image with the owning file and a stage label,
    so the renderer can show which worker is captioning what."""
    from bartleby.commands import scribe as scribe_module
    from bartleby.db.connection import open_db

    pdf = tmp_path / "img.pdf"
    _pdf_with_image(pdf, _png_bytes())

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        writer = scribe_module.Writer(conn)
        parsed = parsers._parse_document(
            pdf, ".pdf", file_hash="h", file_name="img.pdf",
            pdf_converter="pdfplumber", html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_enabled=True, vision_max_dimension=1024, vision_min_dimension=32,
            archive_root=archive_root,
        )
        document_id = writer.persist_parse(parsed)

        lanes: list[tuple[object, str, str]] = []
        caption._caption_all(
            writer,
            [parse.DocUnit(document_id, "img.pdf", "h")],
            vision_provider=_StubVisionProvider(), vision_model="stub-vl:1",
            vision_enabled=True, caption_workers=1, timings=False,
            on_progress=None,
            on_lane=lambda key, item, stage: lanes.append((key, item, stage)),
        )
    finally:
        conn.close()

    assert lanes, "expected a lane update for the one embedded image"
    assert all(item == "img.pdf" and stage == "captioning" for _, item, stage in lanes)


def test_document_id_for_returns_parsed_document(
    isolated_project, tmp_path, mock_embed
):
    """Writer.document_id_for resolves an already-parsed file by its hash."""
    src = _write_txt(tmp_path / "doc.txt", "Some content for ingestion.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        file_hash, doc_id = conn.cursor().execute(
            "SELECT file_hash, document_id FROM documents"
        ).fetchone()
        writer = scribe.Writer(conn)
        assert writer.document_id_for(file_hash) == doc_id
        assert writer.document_id_for("nope") is None
    finally:
        conn.close()


def test_is_complete_text_document(isolated_project, tmp_path, mock_embed):
    """A parsed text doc (chunks, no images) is drain-complete — nothing to caption."""
    src = _write_txt(tmp_path / "doc.txt", "Some content for ingestion.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        doc_id = conn.cursor().execute(
            "SELECT document_id FROM documents"
        ).fetchone()[0]
        writer = scribe.Writer(conn)
        assert classify._is_complete(writer, doc_id)
    finally:
        conn.close()


def test_is_complete_false_when_image_uncaptioned(isolated_project):
    """An image row recorded by parse but not yet captioned (analysis_json
    NULL) keeps the document incomplete — the per-unit resume signal."""
    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("imghash", "image.jpg", "/tmp/image.jpg", None, 0),
        )
        doc_id = conn.last_insert_rowid()
        # analysis_json IS NULL → uncaptioned.
        cur.execute(
            "INSERT INTO images (file_hash, file_path, width, height, "
            "analysis_json, analysis_model) VALUES (?, ?, ?, ?, NULL, NULL)",
            ("imgblob", "/tmp/image.jpg", 100, 100),
        )
        image_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_images "
            "(document_id, image_id, page_number, image_index_on_page) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, image_id, None, 0),
        )
        writer = scribe.Writer(conn)
        assert not classify._is_complete(writer, doc_id)
        assert [pi.image_id for pi in writer.uncaptioned_images(doc_id)] == [image_id]

        # Caption it → complete, and no longer pending.
        cur.execute(
            "UPDATE images SET analysis_json = '{}', analysis_model = 'm' "
            "WHERE image_id = ?", (image_id,),
        )
        assert classify._is_complete(writer, doc_id)
        assert writer.uncaptioned_images(doc_id) == []
    finally:
        conn.close()


def test_documents_needing_summary_filters_summarized_and_empty(isolated_project):
    """The summarize work-list (issue #167) is documents with indexed chunks and
    no summary row; already-summarized docs and zero-chunk docs (nothing
    extractable) are both excluded."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()

        def _add_doc(file_hash, file_name):
            cur.execute(
                "INSERT INTO documents (file_hash, file_name, file_path, "
                "page_count, token_count) VALUES (?, ?, ?, ?, ?)",
                (file_hash, file_name, f"/tmp/{file_name}", None, 0),
            )
            return conn.last_insert_rowid()

        def _chunk():
            return [ChunkInput(
                text="body", embedding=[0.0] * EMBEDDING_DIM, chunk_index=0,
            )]

        # Doc A: has a chunk, no summary → owed.
        doc_a = _add_doc("a", "a.txt")
        insert_document_chunks(conn, doc_a, _chunk())
        # Doc B: has a chunk but already summarized → excluded.
        doc_b = _add_doc("b", "b.txt")
        insert_document_chunks(conn, doc_b, _chunk())
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_b, "t", "d", "body", "m"),
        )
        # Doc C: no chunks at all → excluded (nothing to summarize).
        _add_doc("c", "c.txt")

        owed = scribe.Writer(conn).documents_needing_summary()
        assert [p.document_id for p in owed] == [doc_a]
        assert owed[0].file_name == "a.txt"
        assert owed[0].file_hash == "a"
    finally:
        conn.close()


def test_scribe_does_not_reparse_existing_document(
    isolated_project, tmp_path, mock_embed
):
    """Re-running ingest on an already-complete file is a no-op: no duplicate
    rows, no re-parse, no extra chunks. Parse is atomic, so a document row
    means a finished parse — never a partial to clean up and redo."""
    src = _write_txt(tmp_path / "doc.txt", "Stable content.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        before = (
            cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        )
    finally:
        conn.close()

    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        after = (
            cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        )
        assert after == before == (1, before[1])
    finally:
        conn.close()


def test_scribe_interleaves_image_chunks_into_summary_input(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """The summarizer sees image-chunk text alongside the document body."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic",
            "model": "m",
            "temperature": 0,
            "max_summarize_tokens": 50_000,
            "pdf_converter": "pdfplumber",
            "html_converter": "docling",
            "sparse_text_threshold": 100,
            "ocr_min_confidence": 30,
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    summary_stub = _StubProvider()
    vision_stub = _StubVisionProvider(VlmDescription(
        description="A green rectangle chart with no axis labels.", notes="",
    ))
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: summary_stub if name == "anthropic" else vision_stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    pdf = tmp_path / "doc.pdf"
    body = "Page one prose about the green rectangle figure. " * 6
    _pdf_with_image(pdf, _png_bytes(), text=body)
    scribe.main(project="test_proj", files=str(pdf))

    captured = summary_stub.last_document_text
    assert captured is not None
    # Body text is still there.
    assert "Page one prose" in captured
    # Image-derived chunk text is interleaved in.
    assert "green rectangle chart" in captured
    # And it's labeled so the summarizer can tell where it came from.
    assert "[Image on page 1]" in captured


def test_scribe_persists_authored_date_from_summary(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic", "model": "m",
            "temperature": 0.0, "max_summarize_tokens": 50_000,
        },
    )
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: _StubProvider(
            text="summary body", authored_date="2024-09-12",
        ),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    src = _write_txt(tmp_path / "doc.txt", "Dated document body.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        row = conn.cursor().execute(
            "SELECT authored_date FROM summaries"
        ).fetchone()
        assert row[0] == "2024-09-12"
    finally:
        conn.close()


def test_scribe_drops_malformed_authored_date(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic", "model": "m",
            "temperature": 0.0, "max_summarize_tokens": 50_000,
        },
    )
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: _StubProvider(
            text="summary body", authored_date="Q3 2024",
        ),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    src = _write_txt(tmp_path / "doc.txt", "Document body.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        row = conn.cursor().execute(
            "SELECT authored_date FROM summaries"
        ).fetchone()
        assert row[0] is None
    finally:
        conn.close()


def test_scribe_truncation_note_in_summary(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic",
            "model": "m",
            "temperature": 0,
            "max_summarize_tokens": 5,   # absurdly small so truncation triggers
        },
    )
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: _StubProvider(text="summary body"),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    src = _write_txt(
        tmp_path / "long.txt",
        ("This is a long document. " * 200),  # ~1k tokens, well over 5
    )
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        text = conn.cursor().execute("SELECT text FROM summaries").fetchone()[0]
        assert text.startswith("summary body")
        assert "first 5 tokens" in text
        assert "token document" in text
    finally:
        conn.close()


_EDGAR_SUBMISSION = """\
<SEC-DOCUMENT>0001234567-24-000001.txt : 20240215
<SEC-HEADER>0001234567-24-000001.hdr.sgml : 20240215
CONFORMED SUBMISSION TYPE:\t10-K
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<FILENAME>acme-10k.htm
<TEXT>
<html><body><p>Risk factors and revenue figures.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-31.1
<SEQUENCE>2
<FILENAME>ex31-1.htm
<TEXT>
<html><body><p>Certification.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>3
<FILENAME>ex99-1.txt
<TEXT>
Plain text press release body.
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<SEQUENCE>4
<FILENAME>logo.jpg
<TEXT>
begin 644 logo.jpg
M_]C_X``02D9)
end
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def test_documents_needing_summary_excludes_zero_chunk_document(isolated_project):
    """A document with no indexed chunks is never offered to the summarize pass —
    its only text would be trace garbage that makes the model confabulate (issue
    #80). The work-list excludes it, so it's never handed to the model."""
    conn = open_db("test_proj")
    try:
        conn.cursor().execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("maphash", "annotated-map.pdf", "/tmp/map.pdf", 1, 3),
        )
        # No chunks → the document owes no summarizable work and isn't pending.
        assert scribe.Writer(conn).documents_needing_summary() == []
    finally:
        conn.close()


def test_summary_input_uses_image_chunks_without_doc_chunks(isolated_project):
    """An image-only doc (image chunks, no document chunks) summarizes from
    its real VLM/OCR descriptions — not raw trace text (issue #80)."""
    from bartleby.db.chunks import ChunkInput, insert_image_chunks

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("imgonly", "map.pdf", "/tmp/map.pdf", 1, 3),
        )
        doc_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO images (file_hash, file_path, width, height, "
            "analysis_json, analysis_model) VALUES (?, ?, ?, ?, ?, ?)",
            ("imgblob", "/tmp/map.png", 800, 600, "{}", "m"),
        )
        image_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_images "
            "(document_id, image_id, page_number, image_index_on_page) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, image_id, 2, 0),
        )
        insert_image_chunks(conn, image_id, [ChunkInput(
            text="An annotated railroad track map of the Bellingham subdivision.",
            embedding=[0.1] * EMBEDDING_DIM,
            chunk_index=0,
            content_type="image_description",
        )])

        result = scribe.Writer(conn).summary_input(doc_id)

        # Real image-chunk text is fed, labeled with its page.
        assert "annotated railroad track map" in result
        assert "[Image on page 2]" in result
    finally:
        conn.close()


def test_scribe_summarizes_standalone_image_from_image_chunks(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """A standalone image with a summary provider configured is summarized from
    its image-chunk description (regression: _ingest_image_file no longer
    reconstructs full_text — _build_summary_input reads the chunks directly)."""
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic", "model": "m",
            "temperature": 0.0, "max_summarize_tokens": 50_000,
            "vision_provider": "stub", "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    summary_stub = _StubProvider()
    vision_stub = _StubVisionProvider(VlmDescription(
        description="A photo of a red barn in a field.", notes="",
    ))
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider",
        lambda name, **kwargs: summary_stub if name == "anthropic" else vision_stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    img = tmp_path / "barn.png"
    img.write_bytes(_png_bytes())
    scribe.main(project="test_proj", files=str(img))

    # Summary was produced, and from the image description (not empty input).
    assert summary_stub.calls == 1
    captured = summary_stub.last_document_text
    assert captured is not None
    assert "red barn" in captured
    assert "[Image]" in captured

    conn = open_db("test_proj")
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM summaries"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_scribe_unwraps_edgar_full_submission(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """An EDGAR full-submission `.txt` is detected, unwrapped, and its inner
    HTML routed to sec2md — never to the plain character chunker."""
    from bartleby.ingest.sec2md import Sec2mdChunk, Sec2mdResult

    calls: list[bytes] = []

    def fake_convert_bytes(html: bytes) -> Sec2mdResult:
        calls.append(html)
        return Sec2mdResult(
            full_text="Risk factors and revenue.",
            page_count=1,
            chunks=[
                Sec2mdChunk(text="Risk factors.", section_heading="Item 1A",
                            content_type="sec_text", page_number=1),
                Sec2mdChunk(text="Revenue table.", section_heading=None,
                            content_type="sec_table", page_number=1),
            ],
        )

    monkeypatch.setattr("bartleby.ingest.sec2md.convert_bytes", fake_convert_bytes)

    src = _write_txt(tmp_path / "0001234567-24-000001.txt", _EDGAR_SUBMISSION)
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # One document row for the whole submission.
        assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        # Both inner HTML bodies (10-K + EX-31.1) went through sec2md.
        assert len(calls) == 2

        # 2 chunks per HTML doc + 1 from the plain-text exhibit; graphic skipped.
        n = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0]
        assert n == 5
        assert cur.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == n
        assert cur.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == n

        headings = {
            r[0] for r in cur.execute(
                "SELECT DISTINCT section_heading FROM chunks "
                "WHERE source_kind='document'"
            )
        }
        # sec2md's own header is kept; a missing one falls back to inner TYPE,
        # and the text exhibit is tagged with its TYPE too.
        assert "Item 1A" in headings
        assert "EX-99.1" in headings

        # No raw SGML/tag-soup leaked into any chunk.
        texts = " ".join(
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='document'"
            )
        )
        assert "<SEC-DOCUMENT>" not in texts
        assert "<DOCUMENT>" not in texts
    finally:
        conn.close()


# -------------------- content-sniffed file-type resolution --------------------


def test_resolve_extension_trusts_supported_extension(tmp_path):
    p = _write_txt(tmp_path / "doc.txt", "hi")
    assert resolve_extension(p) == ".txt"


def test_resolve_extension_sniffs_when_extension_missing(tmp_path):
    # The NTSB docket scraper truncates long names, dropping the .PDF entirely.
    pdf = _text_pdf(tmp_path / "ATTACHMENT 7 - LOADING FORMS")
    assert resolve_extension(pdf) == ".pdf"


def test_resolve_extension_sniffs_when_extension_unrecognized(tmp_path):
    # The last dot lands inside "5800.1", so suffix is a junk ".1_ january".
    pdf = _text_pdf(tmp_path / "PHMSA FORMS 5800.1_ JANUARY")
    assert resolve_extension(pdf) == ".pdf"


def test_resolve_extension_does_not_override_recognized_extension(tmp_path):
    # A .txt whose bytes are actually a PDF stays text: no content fallback when
    # the extension is already supported.
    p = _text_pdf(tmp_path / "actually_a_pdf.txt")
    assert resolve_extension(p) == ".txt"


def test_resolve_extension_returns_none_for_unidentifiable(tmp_path):
    p = tmp_path / "mystery"
    p.write_bytes(b"\x00\x01\x02\x03not a known magic signature")
    assert resolve_extension(p) is None


def test_resolve_format_filter_maps_buckets_and_extensions():
    from bartleby.ingest.chunk import resolve_format_filter

    assert resolve_format_filter(["pdf"]) == {".pdf"}
    assert resolve_format_filter(["html"]) == {".html", ".htm"}
    assert resolve_format_filter(["image"]) == {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif",
    }
    # Aliases, bare extensions, and a leading dot all resolve.
    assert resolve_format_filter(["text", "MD", "jpg", ".png"]) == {
        ".txt", ".md", ".jpg", ".png",
    }


def test_resolve_format_filter_rejects_unknown_name():
    from bartleby.ingest.chunk import resolve_format_filter

    with pytest.raises(ValueError):
        resolve_format_filter(["spreadsheet"])


def test_collect_files_single_extensionless_pdf(tmp_path):
    pdf = _text_pdf(tmp_path / "docket_no_ext")
    sources, unidentified = classify._collect_files([pdf])
    assert sources == [(pdf, ".pdf")]
    assert unidentified == []


def test_collect_files_single_unsupported_raises(tmp_path):
    p = tmp_path / "mystery"
    p.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError):
        classify._collect_files([p])


def test_collect_files_directory_sniffs_and_reports_unidentified(tmp_path):
    d = tmp_path / "docket"
    d.mkdir()
    pdf = _text_pdf(d / "ATTACHMENT 7 - LOADING FORMS")     # extensionless PDF
    named = _text_pdf(d / "report.pdf")                     # already has .pdf
    junk = d / "notes"                                      # unidentifiable
    junk.write_bytes(b"\x00\x01\x02\x03")

    sources, unidentified = classify._collect_files([d])

    assert (pdf, ".pdf") in sources
    assert (named, ".pdf") in sources
    assert unidentified == [junk]


def test_collect_files_unions_multiple_directories(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pdf_a = _text_pdf(a / "a.pdf")
    pdf_b = _text_pdf(b / "b.pdf")

    sources, unidentified = classify._collect_files([a, b])

    files = {p for p, _ in sources}
    assert files == {pdf_a, pdf_b}
    assert unidentified == []


def test_collect_files_dedupes_overlapping_roots(tmp_path):
    d = tmp_path / "docket"
    d.mkdir()
    pdf = _text_pdf(d / "report.pdf")

    # The same file reachable both as an explicit file and under its directory.
    sources, _ = classify._collect_files([pdf, d])
    assert sources.count((pdf, ".pdf")) == 1
    assert len(sources) == 1


def test_collect_files_only_filter_restricts_by_resolved_type(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    pdf = _text_pdf(d / "report.pdf")
    txt = _write_txt(d / "notes.txt", "plain text")
    png = d / "image.png"
    png.write_bytes(_png_bytes())

    sources, unidentified = classify._collect_files([d], only={".pdf"})

    assert sources == [(pdf, ".pdf")]
    # txt and png are intentional exclusions — not lumped into unidentified.
    assert unidentified == []
    assert txt not in {p for p, _ in sources}
    assert png not in {p for p, _ in sources}


def test_collect_files_only_filter_matches_sniffed_type(tmp_path):
    d = tmp_path / "docket"
    d.mkdir()
    sniffed = _text_pdf(d / "ATTACHMENT 7 - LOADING FORMS")  # extensionless PDF
    _write_txt(d / "notes.txt", "plain text")

    sources, _ = classify._collect_files([d], only={".pdf"})

    # The content-sniffed PDF is kept by `--only pdf` just like a named one.
    assert sources == [(sniffed, ".pdf")]


def test_scribe_only_filter_end_to_end(isolated_project, tmp_path, mock_embed):
    d = tmp_path / "mixed"
    d.mkdir()
    _text_pdf(d / "report.pdf")
    _write_txt(d / "notes.txt", "Plain text that should be skipped.")

    scribe.main(project="test_proj", files=[str(d)], only=["pdf"])

    conn = open_db("test_proj")
    try:
        names = [
            r[0] for r in conn.cursor().execute("SELECT file_name FROM documents")
        ]
        assert names == ["report.pdf"]
    finally:
        conn.close()


def test_scribe_multiple_paths_end_to_end(isolated_project, tmp_path, mock_embed):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_txt(a / "one.txt", "First document.")
    _write_txt(b / "two.txt", "Second document.")

    scribe.main(project="test_proj", files=[str(a), str(b)])

    conn = open_db("test_proj")
    try:
        names = {
            r[0] for r in conn.cursor().execute("SELECT file_name FROM documents")
        }
        assert names == {"one.txt", "two.txt"}
    finally:
        conn.close()


def test_scribe_ingests_extensionless_pdf_by_sniffing(
    isolated_project, tmp_path, mock_embed
):
    # End to end: a PDF whose filename lost its extension still ingests as a PDF.
    src = _text_pdf(tmp_path / "ATTACHMENT 7 - LOADING FORMS")
    scribe.main(project="test_proj", files=str(src), verbose=False)

    conn = open_db("test_proj")
    try:
        docs = conn.cursor().execute(
            "SELECT file_name, page_count FROM documents"
        ).fetchall()
        assert len(docs) == 1
        assert docs[0][0] == "ATTACHMENT 7 - LOADING FORMS"
        # A page count (txt would be NULL) proves it took the PDF route.
        assert docs[0][1] == 1
    finally:
        conn.close()


# -------------------- per-unit resume + failure tracking --------------------


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

    def analyze_image(self, image_bytes, *, model, media_type="image/jpeg"):
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
    import bartleby.ingest.pdfplumber as pp
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
    scribe.main(project="test_proj", files=str(pdf))
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
    from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS

    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config", _vision_pdf_config,
    )
    vision = _FlakyVisionProvider(fail_times=10_000)  # always fails
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **k: vision,
    )

    pdf = tmp_path / "doc.pdf"
    _pdf_with_image(pdf, _png_bytes(), text="Body text " * 12)

    # Each run makes exactly one more attempt until the cap is reached.
    for _ in range(MAX_INGEST_ATTEMPTS):
        scribe.main(project="test_proj", files=str(pdf))
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

    # A further run is a no-op against the VLM — the unit is capped.
    scribe.main(project="test_proj", files=str(pdf))
    assert vision.calls == MAX_INGEST_ATTEMPTS
    conn = open_db("test_proj")
    try:
        assert conn.cursor().execute(
            "SELECT attempts FROM failed_ingests"
        ).fetchone()[0] == MAX_INGEST_ATTEMPTS               # not bumped past cap
    finally:
        conn.close()


def test_writer_persist_parse_dedupes_shared_image(isolated_project, tmp_path):
    """A second document referencing an already-recorded image reuses the row
    (one images row, two joins) and never resets its caption."""
    from bartleby.ingest.writer import ParsedDocument, ParsedImage, Writer

    def _img(page):
        return ParsedImage(
            hash="shared", archive_path=tmp_path / "i.jpg", width=100, height=80,
            page_number=page, image_index_on_page=1,
        )

    conn = open_db("test_proj")
    try:
        writer = Writer(conn)
        doc_a = writer.persist_parse(ParsedDocument(
            file_hash="a", file_name="a.pdf", archive_path=tmp_path / "a.pdf",
            page_count=1, token_count=1, document_chunks=[], images=[_img(1)],
        ))
        # Caption the shared image as if document A processed it.
        conn.cursor().execute(
            "UPDATE images SET analysis_json='{}', analysis_model='m' "
            "WHERE file_hash='shared'"
        )
        doc_b = writer.persist_parse(ParsedDocument(
            file_hash="b", file_name="b.pdf", archive_path=tmp_path / "b.pdf",
            page_count=1, token_count=1, document_chunks=[], images=[_img(2)],
        ))

        cur = conn.cursor()
        assert cur.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM document_images"
        ).fetchone()[0] == 2
        # The shared image is already captioned, so neither doc sees it pending.
        assert writer.uncaptioned_images(doc_a) == []
        assert writer.uncaptioned_images(doc_b) == []
    finally:
        conn.close()


def test_writer_failure_helpers_record_bump_and_clear(isolated_project):
    """record_failure upserts (bumping attempts), failures() reports them with
    the capped flag, and clear_failure removes the row."""
    from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS, Writer

    conn = open_db("test_proj")
    try:
        writer = Writer(conn)
        assert writer.attempts("h", "parse") == 0
        for _ in range(MAX_INGEST_ATTEMPTS):
            writer.record_failure("h", "doc.pdf", "parse", RuntimeError("boom"))
        assert writer.attempts("h", "parse") == MAX_INGEST_ATTEMPTS

        [failure] = writer.failures()
        assert failure.stage == "parse"
        assert failure.file_name == "doc.pdf"
        assert "boom" in failure.error
        assert failure.capped is True

        writer.clear_failure("h", "parse")
        assert writer.attempts("h", "parse") == 0
        assert writer.failures() == []
    finally:
        conn.close()


def test_report_failures_silent_when_no_failures(isolated_project, monkeypatch):
    """The end-of-run block emits nothing when every unit completed."""
    from bartleby.ingest.writer import Writer

    warnings: list[str] = []
    monkeypatch.setattr(scribe.console, "warn", lambda m: warnings.append(m))

    conn = open_db("test_proj")
    try:
        scribe._report_failures(Writer(conn))
    finally:
        conn.close()
    assert warnings == []


def test_report_failures_warns_with_caps_and_display_limit(
    isolated_project, monkeypatch
):
    """The end-of-run warn block headlines the count + capped wording, flags each
    unit as capped vs will-retry, and truncates the listing at 10 with a pointer."""
    from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS, Writer

    warnings: list[str] = []
    monkeypatch.setattr(scribe.console, "warn", lambda m: warnings.append(m))

    conn = open_db("test_proj")
    try:
        writer = Writer(conn)
        # Two capped units (driven to the cap) + eleven that will retry → 13 total,
        # past the 10-line display limit.
        for _ in range(MAX_INGEST_ATTEMPTS):
            writer.record_failure("capped0", "capped0.pdf", "parse", "boom")
            writer.record_failure("capped1", "capped1.pdf", "caption", "boom")
        for i in range(11):
            writer.record_failure(f"retry{i}", f"retry{i}.pdf", "summary", "later")

        scribe._report_failures(writer)
    finally:
        conn.close()

    blob = "\n".join(warnings)
    # Header: total count, capped subtotal + attempt wording, retry tail.
    assert "13 ingest unit(s) did not complete" in blob
    assert f"2 capped after {MAX_INGEST_ATTEMPTS} attempts" in blob
    assert "the rest will retry on the next run" in blob
    # Per-unit flags, both branches.
    assert "capped — not retried" in blob
    assert f"will retry (attempt 1/{MAX_INGEST_ATTEMPTS})" in blob
    # Display cap: only 10 unit lines, then the "… and N more" pointer.
    unit_lines = [w for w in warnings if w.startswith("  [")]
    assert len(unit_lines) == 10
    assert "… and 3 more" in blob
    assert "bartleby project info" in blob


def test_scribe_timings_emits_aggregate_json(
    isolated_project, tmp_path, mock_embed, capsys
):
    """--timings prints a benchmark summary as JSON to stdout (issue #162)."""
    import json

    _write_txt(tmp_path / "a.txt", "First doc with some words.")
    _text_pdf(tmp_path / "b.pdf")
    scribe.main(project="test_proj", files=str(tmp_path), timings=True)

    out = capsys.readouterr().out
    agg = json.loads(out)
    assert agg["docs"] == 2
    assert agg["pages"] == 1            # only the PDF reports a page count
    assert agg["wall_clock_s"] >= 0
    # parse is the one stage every doc passes through (extracting → parse).
    assert "parse" in agg["stages"]
    assert agg["stages"]["parse"]["total_s"] >= 0


def test_scribe_timings_includes_decoupled_summarize_stage(
    isolated_project, tmp_path, mock_embed, monkeypatch, capsys
):
    """--timings folds the decoupled summarize pass back into each document's
    record: the aggregate carries a `summarize` stage, and the doc count isn't
    inflated by the second pass (one merged record per document, issue #167)."""
    import json

    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "one-shot", "provider": "anthropic",
            "model": "test-model", "temperature": 0.0,
            "max_summarize_tokens": 50_000,
        },
    )
    stub = _StubProvider()
    monkeypatch.setattr(
        "bartleby.ingest.resolve.get_provider", lambda name, **kwargs: stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.summary.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    _write_txt(tmp_path / "a.txt", "First doc with some words.")
    _write_txt(tmp_path / "b.txt", "Second doc with other words.")
    scribe.main(project="test_proj", files=str(tmp_path), timings=True)

    assert stub.calls == 2
    agg = json.loads(capsys.readouterr().out)
    assert agg["docs"] == 2  # one record per doc, not doubled by the pass
    assert "summarize" in agg["stages"]
    assert agg["stages"]["summarize"]["total_s"] >= 0


def test_scribe_without_timings_writes_nothing_to_stdout(
    isolated_project, tmp_path, mock_embed, capsys
):
    """Default path is unchanged: no JSON, clean stdout."""
    _write_txt(tmp_path / "a.txt", "A normal ingest, no timing.")
    scribe.main(project="test_proj", files=str(tmp_path))

    assert capsys.readouterr().out == ""


# ---- _resolve_max_workers (no isolated_project: the real resolver, not the
#      max_workers=1 pin that fixture installs) ----

def _fake_machine(monkeypatch, *, cores: int, free_gb: float):
    from types import SimpleNamespace
    from bartleby.ingest import resolve as scribe_module
    monkeypatch.setattr(scribe_module.os, "cpu_count", lambda: cores)
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: SimpleNamespace(available=int(free_gb * 1024 ** 3)),
    )


def test_resolve_max_workers_timings_forces_one(monkeypatch):
    from bartleby.ingest import resolve as scribe_module
    _fake_machine(monkeypatch, cores=16, free_gb=128)
    assert scribe_module._resolve_max_workers({"max_workers": 8}, timings=True) == 1


def test_resolve_max_workers_auto_is_min_of_cores_and_ram(monkeypatch):
    from bartleby.ingest import resolve as scribe_module
    # RAM-bound: 8 cores but only ~48GB free → 48 // 12.0 = 4 workers.
    _fake_machine(monkeypatch, cores=8, free_gb=48)
    assert scribe_module._resolve_max_workers({}, timings=False) == 4
    # CPU-bound: 16 cores, plenty of RAM → 16 − 2 reserved = 14 workers.
    _fake_machine(monkeypatch, cores=16, free_gb=256)
    assert scribe_module._resolve_max_workers({}, timings=False) == 14


def test_resolve_max_workers_auto_reserves_cores_floored_at_one(monkeypatch):
    from bartleby.ingest import resolve as scribe_module
    # RESERVED_CORES (2) is held back from the auto-pick, never below 1.
    _fake_machine(monkeypatch, cores=4, free_gb=128)
    assert scribe_module._resolve_max_workers({}, timings=False) == 2   # 4 − 2
    _fake_machine(monkeypatch, cores=2, free_gb=128)
    assert scribe_module._resolve_max_workers({}, timings=False) == 1   # 2 − 2 → floor
    _fake_machine(monkeypatch, cores=1, free_gb=128)
    assert scribe_module._resolve_max_workers({}, timings=False) == 1   # 1 − 2 → floor


def test_resolve_max_workers_auto_floors_at_one(monkeypatch):
    from bartleby.ingest import resolve as scribe_module
    _fake_machine(monkeypatch, cores=8, free_gb=1)   # 1 // 12.0 = 0 → floored to 1
    assert scribe_module._resolve_max_workers({}, timings=False) == 1


def test_resolve_max_workers_honors_explicit_value_over_auto_with_warning(monkeypatch):
    from bartleby.ingest import resolve as scribe_module
    _fake_machine(monkeypatch, cores=4, free_gb=128)   # auto = 4 − 2 = 2
    warned: list[str] = []
    monkeypatch.setattr(scribe_module.console, "warn", lambda m: warned.append(m))
    assert scribe_module._resolve_max_workers({"max_workers": 6}, timings=False) == 6
    assert warned and "max_workers=6" in warned[0]


def test_scribe_records_ingest_run_and_stamps_units(
    isolated_project, tmp_path, mock_embed
):
    src = _write_txt(tmp_path / "doc.txt", "A small document to ingest and stamp.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        runs = cur.execute(
            "SELECT run_id, finished_at, config_json FROM ingests"
        ).fetchall()
        assert len(runs) == 1
        run_id, finished_at, config_json = runs[0]
        assert finished_at is not None          # finally-block closed the run
        assert "api_key" not in config_json     # secrets never persisted

        # The document and its chunks carry the producing run.
        assert cur.execute(
            "SELECT ingest_run_id FROM documents"
        ).fetchone()[0] == run_id
        chunk_runs = {
            r[0] for r in cur.execute(
                "SELECT DISTINCT ingest_run_id FROM chunks "
                "WHERE source_kind = 'document'"
            )
        }
        assert chunk_runs == {run_id}
    finally:
        conn.close()


def test_scribe_warns_on_config_drift(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    warnings: list[str] = []
    monkeypatch.setattr(
        "bartleby.commands.scribe.console.warn", lambda m: warnings.append(m)
    )

    configs = [{"pdf_converter": "pdfplumber"}]
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config", lambda: configs[0]
    )

    src = _write_txt(tmp_path / "doc.txt", "A document ingested across two runs.")
    scribe.main(project="test_proj", files=str(src))
    assert not [w for w in warnings if "Config drift" in w]  # no prior → silent

    # Second run under a changed knob → a drift warning, but never a block.
    configs[0] = {"pdf_converter": "docling"}
    scribe.main(project="test_proj", files=str(src))
    drift = [w for w in warnings if "Config drift" in w]
    assert any("pdf_converter" in w for w in drift)

    conn = open_db("test_proj")
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM ingests"
        ).fetchone()[0] == 2
    finally:
        conn.close()
