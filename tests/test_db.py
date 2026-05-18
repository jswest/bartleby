"""Unit tests for bartleby.db (schema, connection, chunks, audit)."""

from __future__ import annotations

import apsw
import pytest

from bartleby.db.audit import log_call
from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_document_chunks,
    insert_finding_chunks,
    insert_summary_chunks,
)
from bartleby.db.connection import init_db, open_db
from bartleby.db.schema import EMBEDDING_DIM, SCHEMA_VERSION


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + i * 0.001 for i in range(EMBEDDING_DIM)]


def _insert_doc(conn, file_hash: str = "h") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (file_hash, file_name, file_path) VALUES (?, ?, ?)",
        (file_hash, "doc.pdf", "/tmp/doc.pdf"),
    )
    return conn.last_insert_rowid()


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr("bartleby.db.connection.PROJECTS_DIR", tmp_path)
    init_db("test_proj")
    c = open_db("test_proj")
    yield c
    c.close()


def test_meta_populated_on_init(conn):
    rows = dict(conn.cursor().execute("SELECT key, value FROM meta"))
    assert rows["schema_version"] == str(SCHEMA_VERSION)
    assert rows["embedding_dim"] == str(EMBEDDING_DIM)
    assert rows["embedding_model"]
    assert rows["sqlite_vec_version"]
    assert rows["bartleby_version"]
    assert rows["created_at"]


def test_all_tables_exist(conn):
    cur = conn.cursor()
    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table')"
        )
    }
    for t in (
        "meta",
        "documents",
        "summaries",
        "sessions",
        "findings",
        "finding_citations",
        "chunks",
        "audit_logs",
        # virtual tables show up as type='table' too
        "chunks_fts",
        "chunks_vec",
    ):
        assert t in tables, f"missing table: {t}"


def test_check_constraint_rejects_bad_source_kind(conn):
    cur = conn.cursor()
    _insert_doc(conn, "abc")
    with pytest.raises(apsw.ConstraintError):
        cur.execute(
            "INSERT INTO chunks "
            "(source_kind, source_id, chunk_index, text) VALUES (?, ?, ?, ?)",
            ("bogus", 1, 0, "x"),
        )


def test_open_db_rejects_missing_project(tmp_path, monkeypatch):
    monkeypatch.setattr("bartleby.db.connection.PROJECTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        open_db("nope")


def test_init_db_refuses_existing(tmp_path, monkeypatch):
    monkeypatch.setattr("bartleby.db.connection.PROJECTS_DIR", tmp_path)
    init_db("foo")
    with pytest.raises(FileExistsError):
        init_db("foo")


def test_insert_document_chunks_keeps_fts_and_vec_in_sync(conn):
    cur = conn.cursor()
    doc_id = _insert_doc(conn, "h1")

    chunks = [
        ChunkInput(
            text="hello world", embedding=_emb(0.0), chunk_index=0,
            section_heading="Intro", content_type="text",
        ),
        ChunkInput(
            text="part two", embedding=_emb(0.1), chunk_index=1,
            section_heading="Method", content_type="text",
        ),
    ]
    ids = insert_document_chunks(conn, doc_id, chunks)
    assert len(ids) == 2 and all(isinstance(i, int) for i in ids)

    assert cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    assert cur.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2
    assert cur.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2

    hits = list(cur.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'hello'"
    ))
    assert hits == [(ids[0],)]


def test_summary_chunks_get_summary_source_kind(conn):
    cur = conn.cursor()
    doc_id = _insert_doc(conn, "h2")
    cur.execute(
        "INSERT INTO summaries (document_id, text, model) VALUES (?, ?, ?)",
        (doc_id, "summary text", "test-model"),
    )
    summary_id = conn.last_insert_rowid()
    insert_summary_chunks(conn, summary_id, [
        ChunkInput(text="s", embedding=_emb(), chunk_index=0),
    ])
    row = cur.execute("SELECT source_kind, source_id FROM chunks").fetchone()
    assert row == ("summary", summary_id)


def test_finding_chunks_get_finding_source_kind(conn):
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (name) VALUES (?)", ("test-session",))
    session_id = conn.last_insert_rowid()
    cur.execute(
        "INSERT INTO findings (session_id, title, body) VALUES (?, ?, ?)",
        (session_id, "t", "b"),
    )
    finding_id = conn.last_insert_rowid()
    insert_finding_chunks(conn, finding_id, [
        ChunkInput(text="f", embedding=_emb(), chunk_index=0),
    ])
    row = cur.execute("SELECT source_kind, source_id FROM chunks").fetchone()
    assert row == ("finding", finding_id)


def test_delete_chunks_for_clears_all_three_tables(conn):
    cur = conn.cursor()
    doc_id = _insert_doc(conn, "h3")
    chunks = [
        ChunkInput(text="t", embedding=_emb(), chunk_index=i) for i in range(3)
    ]
    insert_document_chunks(conn, doc_id, chunks)

    deleted = delete_chunks_for(conn, "document", doc_id)
    assert deleted == 3
    assert cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert cur.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
    assert cur.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0


def test_delete_chunks_for_rejects_bad_source_kind(conn):
    with pytest.raises(ValueError):
        delete_chunks_for(conn, "bogus", 1)


def test_insert_validates_embedding_dimension(conn):
    doc_id = _insert_doc(conn, "h4")
    with pytest.raises(ValueError, match="expected 768"):
        insert_document_chunks(conn, doc_id, [
            ChunkInput(text="x", embedding=[0.0, 0.0], chunk_index=0),
        ])


def test_insert_validates_monotonic_chunk_index(conn):
    doc_id = _insert_doc(conn, "h5")
    with pytest.raises(ValueError, match="strictly increasing"):
        insert_document_chunks(conn, doc_id, [
            ChunkInput(text="x", embedding=_emb(), chunk_index=1),
            ChunkInput(text="y", embedding=_emb(), chunk_index=0),
        ])


def test_insert_validates_empty_text(conn):
    doc_id = _insert_doc(conn, "h6")
    with pytest.raises(ValueError, match="non-empty"):
        insert_document_chunks(conn, doc_id, [
            ChunkInput(text="   ", embedding=_emb(), chunk_index=0),
        ])


def test_unique_constraint_on_chunk_index(conn):
    doc_id = _insert_doc(conn, "h7")
    insert_document_chunks(conn, doc_id, [
        ChunkInput(text="a", embedding=_emb(), chunk_index=0),
    ])
    # Second insert with the same (source_kind, source_id, chunk_index)
    # should trip the UNIQUE constraint at the DB layer.
    with pytest.raises(apsw.ConstraintError):
        insert_document_chunks(conn, doc_id, [
            ChunkInput(text="b", embedding=_emb(), chunk_index=0),
        ])


def test_empty_chunks_list_is_a_noop(conn):
    doc_id = _insert_doc(conn, "h8")
    assert insert_document_chunks(conn, doc_id, []) == []


def test_audit_log_insert(conn):
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (name) VALUES (?)", ("s1",))
    sid = conn.last_insert_rowid()

    audit_id = log_call(
        conn,
        session_id=sid,
        tool_name="search",
        args={"query": "hello", "limit": 10},
        result_summary="5 hits",
        duration_ms=42,
    )
    row = cur.execute(
        "SELECT tool_name, args_json, result_summary, duration_ms, session_id "
        "FROM audit_logs WHERE audit_log_id = ?",
        (audit_id,),
    ).fetchone()
    assert row[0] == "search"
    assert row[1] == '{"query": "hello", "limit": 10}'
    assert row[2] == "5 hits"
    assert row[3] == 42
    assert row[4] == sid


def test_audit_log_session_id_nullable(conn):
    audit_id = log_call(
        conn,
        session_id=None,
        tool_name="list_documents",
    )
    row = conn.cursor().execute(
        "SELECT session_id, args_json, duration_ms FROM audit_logs WHERE audit_log_id = ?",
        (audit_id,),
    ).fetchone()
    assert row == (None, None, None)


def test_audit_log_set_null_on_session_delete(conn):
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (name) VALUES (?)", ("s2",))
    sid = conn.last_insert_rowid()
    log_call(conn, session_id=sid, tool_name="x")
    cur.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    row = cur.execute("SELECT session_id FROM audit_logs").fetchone()
    assert row == (None,)
