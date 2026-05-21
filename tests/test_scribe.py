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
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
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
    ):
        self.text = text
        self.title = title
        self.description = description
        self.calls = 0

    def summarize(self, document_text, *, model, temperature):
        self.calls += 1
        return DocumentSummary(
            title=self.title, description=self.description, text=self.text,
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

    bartleby.project.create_project("test_proj")
    yield projects


@pytest.fixture
def mock_embed(monkeypatch):
    """Replace embed_texts with a deterministic stub returning correctly-sized vectors."""
    def fake(texts):
        return _emb(0.0, len(texts))
    # Patch every import site (commands.scribe imports it directly).
    monkeypatch.setattr("bartleby.commands.scribe.embed_texts", fake)
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


def _png_bytes(width=100, height=60, color=(20, 200, 50)) -> bytes:
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
        "bartleby.commands.scribe.get_provider",
        lambda name, **kwargs: stub,
    )
    # Bypass docling for the summary chunking too — return one chunk.
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.commands.scribe.chunk_markdown_string",
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


def test_scribe_ingests_pdf_via_pdfplumber_with_embedded_image(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "backend": "pdfplumber",
            "sparse_text_threshold": 100,
            "ocr_min_confidence": 30,
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.commands.scribe.get_provider",
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


def test_scribe_dedupes_identical_images_across_documents(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        "bartleby.commands.scribe.load_config",
        lambda: {
            "summary_depth": "none",
            "backend": "pdfplumber",
            "sparse_text_threshold": 100,
            "vision_provider": "stub",
            "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    vision = _StubVisionProvider()
    monkeypatch.setattr(
        "bartleby.commands.scribe.get_provider",
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
        "bartleby.commands.scribe.get_provider",
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
            "backend": "pdfplumber",
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


def test_scribe_stage_callback_progresses_through_phases(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """on_stage fires through extracting → embedding → analyzing → summarizing."""
    from bartleby.commands import scribe as scribe_module
    from bartleby.db.connection import open_db

    stub_summary = _StubProvider()
    monkeypatch.setattr(
        scribe_module, "load_config",
        lambda: {
            "summary_depth": "one-shot",
            "provider": "anthropic", "model": "m",
            "temperature": 0.0, "max_summarize_tokens": 50_000,
            "backend": "pdfplumber",
            "sparse_text_threshold": 100, "ocr_min_confidence": 30,
            "vision_provider": "stub", "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    monkeypatch.setattr(
        scribe_module, "get_provider",
        lambda name, **kwargs: stub_summary if name == "anthropic" else _StubVisionProvider(),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        scribe_module, "chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )

    pdf = tmp_path / "img.pdf"
    _pdf_with_image(pdf, _png_bytes())

    stages: list[str] = []
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        scribe_module._process_one(
            conn, pdf, archive_root,
            backend="pdfplumber",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_max_dimension=1024,
            llm_provider=stub_summary, llm_model="m",
            temperature=0.0, max_summarize_tokens=50_000,
            vision_provider=_StubVisionProvider(), vision_model="stub-vl:1",
            on_stage=stages.append,
        )
    finally:
        conn.close()

    # Every phase fires in order; "extracting" is set by _process_one before
    # the pdfplumber dispatcher, then the dispatcher fires embedding +
    # analyzing images, then _process_one tops it off with summarizing.
    assert stages[0] == "extracting"
    assert "embedding" in stages
    assert "analyzing images" in stages
    assert stages[-1] == "summarizing"


def test_scribe_image_progress_callback_fires_per_image(
    isolated_project, tmp_path, mock_embed, monkeypatch
):
    """_run_image_routes invokes on_progress(0, N) then once per image."""
    from bartleby.commands import scribe as scribe_module
    from bartleby.db.connection import open_db

    monkeypatch.setattr(
        scribe_module, "load_config",
        lambda: {
            "summary_depth": "none",
            "backend": "pdfplumber",
            "sparse_text_threshold": 100,
            "vision_provider": "stub", "vision_model": "stub-vl:1",
            "vision_max_dimension": 1024,
        },
    )
    monkeypatch.setattr(
        scribe_module, "get_provider",
        lambda name, **kwargs: _StubVisionProvider(),
    )

    pdf = tmp_path / "img.pdf"
    _pdf_with_image(pdf, _png_bytes())

    seen: list[tuple[int, int]] = []
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = open_db("test_proj")
    try:
        scribe_module._process_one(
            conn, pdf, archive_root,
            backend="pdfplumber",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_max_dimension=1024,
            llm_provider=None, llm_model=None,
            temperature=0.0, max_summarize_tokens=1000,
            vision_provider=_StubVisionProvider(), vision_model="stub-vl:1",
            on_image_progress=lambda done, total: seen.append((done, total)),
        )
    finally:
        conn.close()

    assert seen, "expected at least one progress callback for one embedded image"
    assert seen[0] == (0, 1)
    assert seen[-1] == (1, 1)


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
        "bartleby.commands.scribe.get_provider",
        lambda name, **kwargs: _StubProvider(text="summary body"),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.commands.scribe.chunk_markdown_string",
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
