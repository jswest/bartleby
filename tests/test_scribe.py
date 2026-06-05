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

    def summarize(self, document_text, *, model, temperature):
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
        "bartleby.commands.scribe.get_provider",
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
            "pdf_converter": "pdfplumber",
            "html_converter": "docling",
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
            conn, pdf, ".pdf", archive_root,
            pdf_converter="pdfplumber",
            html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_max_dimension=1024, vision_min_dimension=32,
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
            "pdf_converter": "pdfplumber",
            "html_converter": "docling",
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
            conn, pdf, ".pdf", archive_root,
            pdf_converter="pdfplumber",
            html_converter="docling",
            sparse_text_threshold=100, ocr_min_confidence=30,
            vision_max_dimension=1024, vision_min_dimension=32,
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


def test_document_already_ingested_skips_partial_row(isolated_project):
    """A documents row with no chunks/images is treated as not-yet-ingested."""
    conn = open_db("test_proj")
    try:
        conn.cursor().execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("abc123", "stranded.txt", "/tmp/stranded.txt", None, 0),
        )
        assert scribe._document_already_ingested(conn, "abc123") is None
    finally:
        conn.close()


def test_document_already_ingested_recognizes_complete_text_doc(
    isolated_project, tmp_path, mock_embed
):
    """A document with at least one text chunk is recognized as complete."""
    src = _write_txt(tmp_path / "doc.txt", "Some content for ingestion.")
    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        file_hash, doc_id = conn.cursor().execute(
            "SELECT file_hash, document_id FROM documents"
        ).fetchone()
        assert scribe._document_already_ingested(conn, file_hash) == doc_id
    finally:
        conn.close()


def test_document_already_ingested_recognizes_image_only_doc(isolated_project):
    """A document with only document_images attached (no doc chunks) is complete."""
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
        cur.execute(
            "INSERT INTO images (file_hash, file_path, width, height, "
            "analysis_json, analysis_model) VALUES (?, ?, ?, ?, ?, ?)",
            ("imgblob", "/tmp/image.jpg", 100, 100, "{}", "m"),
        )
        image_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_images "
            "(document_id, image_id, page_number, image_index_on_page) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, image_id, None, 0),
        )
        assert scribe._document_already_ingested(conn, "imghash") == doc_id
    finally:
        conn.close()


def test_scribe_reingests_after_partial_crash(
    isolated_project, tmp_path, mock_embed
):
    """A stranded documents row from a crashed run is cleaned up + re-ingested."""
    src = _write_txt(tmp_path / "doc.txt", "Content that should land on retry.")
    # Simulate an interrupted previous run: hash matches the file but no
    # chunks were ever written. The row holds the file_hash UNIQUE slot.
    file_hash = scribe._hash_file(src)
    conn = open_db("test_proj")
    try:
        # The stranded row's file_path points at a non-existent location and
        # its token_count is 0 — the canonical "partial" fingerprint.
        conn.cursor().execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_hash, "doc.txt", "/tmp/missing.txt", None, 0),
        )
    finally:
        conn.close()

    scribe.main(project="test_proj", files=str(src))

    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # Exactly one row for this hash, and it now looks like a real ingest:
        # the stranded "/tmp/missing.txt" path was replaced with a real archive
        # path and tokens were counted.
        rows = cur.execute(
            "SELECT file_path, token_count FROM documents WHERE file_hash = ?",
            (file_hash,),
        ).fetchall()
        assert len(rows) == 1
        archive_path, token_count = rows[0]
        assert archive_path != "/tmp/missing.txt"
        assert token_count > 0
        # Chunks landed this time.
        n_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='document'"
        ).fetchone()[0]
        assert n_chunks >= 1
    finally:
        conn.close()


def test_cleanup_partial_document_preserves_shared_image(isolated_project):
    """Cleaning up a partial doc drops its joins but leaves shared images alive."""
    conn = open_db("test_proj")
    try:
        cur = conn.cursor()
        # Two documents share one image
        for fh, name in (("h_partial", "partial.pdf"), ("h_other", "other.pdf")):
            cur.execute(
                "INSERT INTO documents "
                "(file_hash, file_name, file_path, page_count, token_count) "
                "VALUES (?, ?, ?, 1, 0)", (fh, name, f"/tmp/{name}"),
            )
        partial_id, other_id = [r[0] for r in cur.execute(
            "SELECT document_id FROM documents ORDER BY document_id"
        )]
        cur.execute(
            "INSERT INTO images (file_hash, file_path, width, height, "
            "analysis_json, analysis_model) VALUES (?, ?, ?, ?, ?, ?)",
            ("img_shared", "/tmp/img.jpg", 100, 100, "{}", "m"),
        )
        image_id = conn.last_insert_rowid()
        for doc_id in (partial_id, other_id):
            cur.execute(
                "INSERT INTO document_images "
                "(document_id, image_id, page_number, image_index_on_page) "
                "VALUES (?, ?, NULL, 0)", (doc_id, image_id),
            )

        scribe._cleanup_partial_document(conn, partial_id)

        # Partial doc + its join row are gone
        assert cur.execute(
            "SELECT COUNT(*) FROM documents WHERE document_id = ?",
            (partial_id,),
        ).fetchone()[0] == 0
        assert cur.execute(
            "SELECT COUNT(*) FROM document_images WHERE document_id = ?",
            (partial_id,),
        ).fetchone()[0] == 0
        # Other doc + shared image survive
        assert cur.execute(
            "SELECT COUNT(*) FROM documents WHERE document_id = ?",
            (other_id,),
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM images WHERE image_id = ?",
            (image_id,),
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM document_images WHERE document_id = ?",
            (other_id,),
        ).fetchone()[0] == 1
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
        "bartleby.commands.scribe.get_provider",
        lambda name, **kwargs: summary_stub if name == "anthropic" else vision_stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.commands.scribe.chunk_markdown_string",
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
        "bartleby.commands.scribe.get_provider",
        lambda name, **kwargs: _StubProvider(
            text="summary body", authored_date="2024-09-12",
        ),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.commands.scribe.chunk_markdown_string",
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
        "bartleby.commands.scribe.get_provider",
        lambda name, **kwargs: _StubProvider(
            text="summary body", authored_date="Q3 2024",
        ),
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.commands.scribe.chunk_markdown_string",
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


def test_maybe_summarize_skips_zero_chunk_document(isolated_project):
    """A document with no indexed chunks is never handed to the summarizer —
    its only text is `full_text` trace garbage that makes the model
    confabulate (issue #80)."""
    conn = open_db("test_proj")
    try:
        conn.cursor().execute(
            "INSERT INTO documents "
            "(file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("maphash", "annotated-map.pdf", "/tmp/map.pdf", 1, 3),
        )
        doc_id = conn.last_insert_rowid()

        stub = _StubProvider()
        # full_text is the kind of trace string that defeats the old
        # whitespace-only guard: non-empty after .strip().
        scribe._maybe_summarize(
            conn, doc_id, "Map\n\f\nMap",
            provider=stub, model="m",
            temperature=0.0, max_summarize_tokens=1000,
        )

        # Provider never called, nothing persisted.
        assert stub.calls == 0
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM summaries"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_build_summary_input_uses_image_chunks_without_doc_chunks(
    isolated_project,
):
    """An image-only doc (image chunks, no document chunks) summarizes from
    its real VLM/OCR descriptions — not the trace `full_text` (issue #80)."""
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

        result = scribe._build_summary_input(conn, doc_id, "Map\n\f\nMap")

        # Real image-chunk text is fed, labeled with its page; trace is dropped.
        assert "annotated railroad track map" in result
        assert "[Image on page 2]" in result
        assert "Map\n\f\nMap" not in result
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
        "bartleby.commands.scribe.get_provider",
        lambda name, **kwargs: summary_stub if name == "anthropic" else vision_stub,
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.commands.scribe.chunk_markdown_string",
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


def test_collect_files_single_extensionless_pdf(tmp_path):
    pdf = _text_pdf(tmp_path / "docket_no_ext")
    sources, unidentified = scribe._collect_files(pdf)
    assert sources == [(pdf, ".pdf")]
    assert unidentified == []


def test_collect_files_single_unsupported_raises(tmp_path):
    p = tmp_path / "mystery"
    p.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError):
        scribe._collect_files(p)


def test_collect_files_directory_sniffs_and_reports_unidentified(tmp_path):
    d = tmp_path / "docket"
    d.mkdir()
    pdf = _text_pdf(d / "ATTACHMENT 7 - LOADING FORMS")     # extensionless PDF
    named = _text_pdf(d / "report.pdf")                     # already has .pdf
    junk = d / "notes"                                      # unidentifiable
    junk.write_bytes(b"\x00\x01\x02\x03")

    sources, unidentified = scribe._collect_files(d)

    assert (pdf, ".pdf") in sources
    assert (named, ".pdf") in sources
    assert unidentified == [junk]


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
