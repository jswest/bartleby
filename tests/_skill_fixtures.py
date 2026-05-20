"""Shared fixtures and helpers for skill-script tests."""

from __future__ import annotations

import pytest

import bartleby.config
import bartleby.db.connection
import bartleby.project
from bartleby.db.chunks import (
    ChunkInput,
    insert_document_chunks,
    insert_image_chunks,
)
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + 0.001 * j for j in range(EMBEDDING_DIM)]


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(bartleby.config, "BARTLEBY_DIR", tmp_path)
    monkeypatch.setattr(bartleby.config, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(bartleby.project, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.db.connection, "PROJECTS_DIR", projects)
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

        # Summary for doc A
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_a, "Alpha", "Test summary of alpha document.",
             "A summary of alpha.", "test"),
        )
    finally:
        conn.close()

    return {"project": project_env, "doc_a": doc_a, "doc_b": doc_b}


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
