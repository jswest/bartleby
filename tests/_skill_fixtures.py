"""Shared fixtures and helpers for skill-script tests."""

from __future__ import annotations

import pytest

import bartleby.project
from bartleby.db.chunks import (
    ChunkInput,
    insert_document_chunks,
    insert_finding_chunks,
    insert_image_chunks,
    insert_summary_chunks,
)
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + 0.001 * j for j in range(EMBEDDING_DIM)]


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    """Autouse stub so finding/summary skill tests never reach the real BGE model.

    Patches the names ``skill_scripts._common`` looks up: a deterministic
    ``embed_texts`` and a single-``ChunkRow`` ``chunk_markdown_string`` (the
    latter keeps chunk_count assertions stable). Only active in modules that
    import it.
    """
    monkeypatch.setattr(
        "bartleby.ingest.embed.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.chunk.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def seed_finding(conn, session_id, *, title="A finding", description="hook",
                 body="body", cited_chunk_ids=()) -> tuple[int, list[int]]:
    """Insert a finding (+ one body chunk, + optional citations) directly.

    Returns ``(finding_id, body_chunk_ids)``. The body chunk goes through the
    typed ``insert_finding_chunks`` helper (chunks discipline).
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, title, description, body),
    )
    finding_id = conn.last_insert_rowid()
    body_chunk_ids = insert_finding_chunks(conn, finding_id, [
        ChunkInput(text=body, embedding=_emb(), chunk_index=0),
    ])
    for chunk_id in cited_chunk_ids:
        cur.execute(
            "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
            (finding_id, chunk_id),
        )
    return finding_id, body_chunk_ids


def assert_chunk_tables_consistent(conn) -> None:
    """Assert ``chunks``, ``chunks_fts``, and ``chunks_vec`` agree.

    The two mirror tables are checked DIFFERENTLY because they are different
    kinds of virtual table:

    - ``chunks_fts`` is an external-content FTS5 table (``content='chunks'``),
      so any non-MATCH read (rowid lookup, ``COUNT``) is satisfied THROUGH
      ``chunks`` and would be vacuous. The only way to catch drift between the
      FTS index and its content table is FTS5's own ``'integrity-check'``
      command. It MUST be invoked in the two-argument ``rank=1`` form
      (``VALUES('integrity-check', 1)``): the rank argument is what makes
      integrity-check actually re-derive the index from the external-content
      ``chunks`` table and compare them, raising on drift in either direction.
      The one-argument form is a no-op for content/index drift in this SQLite
      (apsw 3.51 / SQLite 3.51) — it passes even when the index is missing rows
      or holds stale entries.
    - ``chunks_vec`` is a real ``vec0`` table with its own row storage, so its
      ``rowid`` set can be compared against ``chunks`` directly.
    """
    cur = conn.cursor()
    # FTS leg: external-content integrity check (rank=1 raises on drift).
    cur.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)")
    # Vector leg: rowid sets must match exactly.
    chunk_ids = {r[0] for r in cur.execute("SELECT chunk_id FROM chunks")}
    vec_ids = {r[0] for r in cur.execute("SELECT rowid FROM chunks_vec")}
    assert vec_ids == chunk_ids, (
        f"chunks_vec rowids drifted from chunks: "
        f"only in vec={sorted(vec_ids - chunk_ids)}, "
        f"only in chunks={sorted(chunk_ids - vec_ids)}"
    )


@pytest.fixture
def project_env():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    bartleby.project.create_project("p")
    yield "p"


@pytest.fixture
def seeded_project(project_env):
    """A project with two documents (one with a summary) and an active session."""
    conn = open_db(project_env)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("h1", "alpha.pdf", "/tmp/alpha.pdf", 5, 1000),
        )
        doc_a = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("h2", "beta.txt", "/tmp/beta.txt", None, 200),
        )
        doc_b = conn.last_insert_rowid()

        # Document A: 4 chunks
        insert_document_chunks(conn, doc_a, [
            ChunkInput(text="alpha chunk zero about pm25", embedding=_emb(0.0),
                       chunk_index=0, section_heading="Intro", content_type="text"),
            ChunkInput(text="alpha chunk one about equity", embedding=_emb(0.1),
                       chunk_index=1, section_heading="Methods", content_type="text"),
            ChunkInput(text="alpha chunk two on results", embedding=_emb(0.2),
                       chunk_index=2, section_heading="Results", content_type="text"),
            ChunkInput(text="alpha chunk three concludes", embedding=_emb(0.3),
                       chunk_index=3, section_heading="Conclusion", content_type="text"),
        ])

        # Document B: 2 chunks
        insert_document_chunks(conn, doc_b, [
            ChunkInput(text="beta chunk zero says hello", embedding=_emb(1.0),
                       chunk_index=0),
            ChunkInput(text="beta chunk one says farewell", embedding=_emb(1.1),
                       chunk_index=1),
        ])

        # Summary for doc A — seeded WITH chunks so a later replace has prior
        # chunks to delete from all three tables (the save_summary replace test
        # asserts they are gone afterward).
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_a, "Alpha", "Test summary of alpha document.",
             "A summary of alpha.", "test"),
        )
        summary_a = conn.last_insert_rowid()
        summary_a_chunk_ids = insert_summary_chunks(conn, summary_a, [
            ChunkInput(text="alpha summary chunk zero", embedding=_emb(3.0),
                       chunk_index=0),
            ChunkInput(text="alpha summary chunk one", embedding=_emb(3.1),
                       chunk_index=1),
        ])
    finally:
        conn.close()

    return {
        "project": project_env,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "summary_a": summary_a,
        "summary_a_chunk_ids": summary_a_chunk_ids,
    }


def seed_image(conn, document_id: int, *, file_hash: str, file_path: str,
               description: str = "A test image scene.",
               ocr_text: str = "WELCOME",
               page_number: int | None = 1,
               image_index_on_page: int = 1) -> int:
    """Helper for image-scope tests: insert an image, link it to a doc, chunk it."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO images "
        "(file_hash, file_path, width, height, analysis_json, analysis_model) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_hash, file_path, 800, 600,
         '{"kind":"scene","text":"' + ocr_text + '","description":"' + description + '","notes":""}',
         "fake-vlm"),
    )
    image_id = conn.last_insert_rowid()
    cur.execute(
        "INSERT INTO document_images "
        "(document_id, image_id, page_number, image_index_on_page) "
        "VALUES (?, ?, ?, ?)",
        (document_id, image_id, page_number, image_index_on_page),
    )
    insert_image_chunks(conn, image_id, [
        ChunkInput(text=ocr_text, embedding=_emb(2.0), chunk_index=0,
                   content_type="image_ocr"),
        ChunkInput(text=description, embedding=_emb(2.1), chunk_index=1,
                   content_type="image_description"),
    ])
    return image_id
