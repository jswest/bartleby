"""Smoke test for skill/scripts/delete_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import delete_finding, save_finding
from bartleby.db.chunks import ChunkInput, insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.session import start_session
from tests._skill_fixtures import (  # noqa: F401
    assert_chunk_tables_consistent,
    project_env,
    seeded_project,
)


def _seed_finding_as(conn, session_id, title="Foreign draft") -> int:
    """Insert a minimal finding owned by ``session_id``; return its id."""
    conn.cursor().execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, title, "hook", "body"),
    )
    finding_id = conn.last_insert_rowid()
    insert_finding_chunks(conn, finding_id, [
        ChunkInput(
            text="body",
            embedding=[0.01 * i for i in range(EMBEDDING_DIM)],
            chunk_index=0,
        ),
    ])
    return finding_id


def _finding_exists(project, finding_id) -> bool:
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id = ?", (finding_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr(
        "bartleby.ingest.embed.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.chunk.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def _seed_finding(seeded_project, tmp_path, capsys) -> dict:
    """Save a baseline finding citing two document chunks; return the response."""
    conn = open_db(seeded_project["project"])
    try:
        cited = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 2",
            (seeded_project["doc_a"],),
        ).fetchall()
        a, b = (r[0] for r in cited)
    finally:
        conn.close()

    body_file = tmp_path / "f.md"
    body_file.write_text(f"# F\n\nClaim[^{a}]. Two[^{b}].", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "Stale draft",
        "--description", "A draft to retract.",
        "--body-file", str(body_file),
    ])
    saved = json.loads(capsys.readouterr().out)
    saved["_chunks"] = (a, b)
    return saved


def test_delete_finding_removes_row_chunks_and_citations(
    seeded_project, tmp_path, capsys
):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]
    a, b = saved["_chunks"]

    # Capture the finding's own body chunk ids before deletion.
    conn = open_db(seeded_project["project"])
    try:
        body_chunk_ids = [
            r[0] for r in conn.cursor().execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ?",
                (finding_id,),
            )
        ]
    finally:
        conn.close()
    assert body_chunk_ids  # sanity: the finding had body chunks

    delete_finding.main([
        "--project", seeded_project["project"],
        "--finding", str(finding_id),
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["status"] == "deleted"
    assert out["finding_id"] == finding_id
    assert out["title"] == "Stale draft"
    assert out["removed_chunks"] == len(body_chunk_ids)
    assert out["removed_citations"] == 2

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # The findings row is gone.
        assert cur.execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id = ?", (finding_id,),
        ).fetchone()[0] == 0
        # finding_citations cascaded.
        assert cur.execute(
            "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()[0] == 0
        # The finding's body chunks are gone from chunks + the vec mirror.
        # (chunks_fts is external-content, so a per-rowid COUNT reads THROUGH
        # chunks and is vacuous — the FTS leg is verified by the integrity
        # check in assert_chunk_tables_consistent below instead.)
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding' "
            "AND source_id = ?",
            (finding_id,),
        ).fetchone()[0] == 0
        for cid in body_chunk_ids:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,)
            ).fetchone()[0] == 0
        # The cited *document* chunks (evidence) are untouched.
        for cid in (a, b):
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_id = ?", (cid,)
            ).fetchone()[0] == 1

        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


def test_delete_finding_unknown_id(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        delete_finding.main([
            "--project", seeded_project["project"],
            "--finding", "99999",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "FINDING_NOT_FOUND"


def test_delete_finding_memory_off_other_session(seeded_project, capsys):
    """A memory-off session cannot delete a finding another session authored,
    and the foreign finding survives the rejected call."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        finding_id = _seed_finding_as(conn, author)
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    with pytest.raises(SystemExit) as exc:
        delete_finding.main([
            "--project", project, "--finding", str(finding_id),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"
    # The gate fires before any delete — the finding is untouched.
    assert _finding_exists(project, finding_id)


def test_delete_finding_memory_off_own_session(seeded_project, capsys):
    """A memory-off session can still delete a finding it authored itself."""
    project = seeded_project["project"]
    info = start_session(project, memory_enabled=False)

    conn = open_db(project)
    try:
        finding_id = _seed_finding_as(conn, info["session_id"], title="own")
    finally:
        conn.close()

    delete_finding.main(["--project", project, "--finding", str(finding_id)])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "deleted"
    assert out["finding_id"] == finding_id
    assert not _finding_exists(project, finding_id)
