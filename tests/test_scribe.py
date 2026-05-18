"""End-to-end ingest test for `.txt` with mocked embedder and provider."""

from __future__ import annotations

import pytest

import bartleby.config
import bartleby.db.connection
import bartleby.project
from bartleby.commands import scribe
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.providers.base import DocumentSummary


def _emb(seed: float, n: int) -> list[list[float]]:
    return [
        [seed + 0.0001 * i for _ in range(EMBEDDING_DIM)] for i in range(n)
    ]


class _StubProvider:
    name = "stub"

    def __init__(self, text: str = "## Summary\n\nA stub summary."):
        self.text = text
        self.calls = 0

    def summarize(self, document_text, *, model, temperature):
        self.calls += 1
        return DocumentSummary(text=self.text)


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
    return fake


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
            "SELECT text, model FROM summaries"
        ).fetchall()
        assert len(summaries) == 1
        assert summaries[0][0].startswith("## Stub summary")
        assert summaries[0][1] == "test-model"

        sum_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='summary'"
        ).fetchone()[0]
        assert sum_chunks == 1
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
