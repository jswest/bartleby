"""`bartleby project info --verify` — read-only corpus integrity audit (#487).

A positive (all-pass) control plus the failure case the issue calls for: a
deliberately-orphaned chunk that the ``no_orphan_chunks`` check must catch
(non-zero exit, the failing check named). Exercises the command through
``project_cmd.info(..., verify=True)``, the same entrypoint the CLI uses.
"""

from __future__ import annotations

import io
import struct

import apsw
import pytest
from rich.console import Console

import bartleby.project
from bartleby.commands import project as project_cmd
from bartleby.db.chunks import ChunkInput, insert_document_chunks
from bartleby.db.connection import _attach, open_db, project_db_path
from bartleby.db.schema import EMBEDDING_DIM


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + i * 0.001 for i in range(EMBEDDING_DIM)]


@pytest.fixture
def seeded(tmp_path):
    """A project with one document + two chunks (a structurally-sound corpus)."""
    bartleby.project.create_project("alpha")
    conn = open_db("alpha")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path) VALUES (?, ?, ?)",
            ("h", "doc.pdf", "/tmp/doc.pdf"),
        )
        doc_id = conn.last_insert_rowid()
        insert_document_chunks(conn, doc_id, [
            ChunkInput(text="alpha one", embedding=_emb(0.0), chunk_index=0),
            ChunkInput(text="alpha two", embedding=_emb(0.1), chunk_index=1),
        ])
    finally:
        conn.close()
    return "alpha"


def _run_verify(name: str, monkeypatch) -> tuple[str, int | None]:
    """Run ``info(..., verify=True)``, returning (rendered output, exit code).

    Exit code is ``None`` when the command did not call ``sys.exit`` (all pass).
    """
    buf = io.StringIO()
    monkeypatch.setattr(project_cmd, "_console", Console(file=buf, width=200))
    code: int | None = None
    try:
        project_cmd.info(name=name, verify=True)
    except SystemExit as e:
        code = e.code if e.code is not None else 0
    return buf.getvalue(), code


def test_verify_passes_on_sound_corpus(seeded, monkeypatch):
    out, code = _run_verify(seeded, monkeypatch)
    assert code is None, f"expected all-pass (no exit), got exit {code}\n{out}"
    assert "Integrity checks" in out
    # Every check reports PASS; none FAIL.
    for check in (
        "tri_table_sync",
        "no_orphan_chunks",
        "schema_matches_stamp",
        "failed_ingests_sanity",
    ):
        assert check in out
    assert "FAIL" not in out
    assert "PASS" in out


def test_verify_catches_orphaned_chunk(seeded, monkeypatch):
    """A chunk whose source_id has no parent document fails the audit non-zero.

    Seeds the orphan into all three tables (chunks + the fts/vec mirrors) so the
    tri-table-sync check stays green and the failure isolates to no_orphan_chunks.
    """
    db_path = project_db_path(seeded)
    conn = apsw.Connection(str(db_path))
    try:
        _attach(conn)
        cur = conn.cursor()
        with conn:
            cur.execute(
                "INSERT INTO chunks "
                "(source_kind, source_id, chunk_index, text) "
                "VALUES ('document', 999999, 0, 'orphaned chunk')"
            )
            cid = conn.last_insert_rowid()
            cur.execute(
                "INSERT INTO chunks_fts(rowid, text, section_heading) "
                "VALUES (?, 'orphaned chunk', '')",
                (cid,),
            )
            cur.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                (cid, struct.pack(f"{EMBEDDING_DIM}f", *_emb(9.0))),
            )
    finally:
        conn.close()

    out, code = _run_verify(seeded, monkeypatch)
    assert code == 1, f"expected non-zero exit on orphan, got {code}\n{out}"
    assert "FAIL" in out
    assert "no_orphan_chunks" in out
    # The other checks still report (the audit runs the full battery).
    assert "tri_table_sync" in out


def test_verify_survives_dropped_chunks_vec(seeded, monkeypatch):
    """A corrupt DB (``chunks_vec`` dropped) reports a FAIL, never a traceback.

    ``check_tri_table_sync``'s ``SELECT rowid FROM chunks_vec`` raises a raw
    ``apsw.SQLError`` once the table is gone. The per-check guard in
    ``run_all_checks`` must turn that into a FAILED result and let the remaining
    checks still run, so ``--verify`` exits non-zero with a FAIL line rather than
    dumping a traceback.
    """
    db_path = project_db_path(seeded)
    conn = apsw.Connection(str(db_path))
    try:
        _attach(conn)
        with conn:
            conn.cursor().execute("DROP TABLE chunks_vec")
    finally:
        conn.close()

    out, code = _run_verify(seeded, monkeypatch)
    assert code == 1, f"expected non-zero exit on dropped vec, got {code}\n{out}"
    assert "FAIL" in out
    # The check that hit the missing table is reported as a failure...
    assert "tri_table_sync" in out
    # ...and the battery still ran the later checks past the failing one.
    assert "failed_ingests_sanity" in out
    # No raw traceback leaked into the rendered output.
    assert "Traceback" not in out
