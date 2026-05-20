"""Smoke test for skill/scripts/read_chunks.py."""

from __future__ import annotations

import json

import pytest

from bartleby.db.connection import open_db
from bartleby.skill_scripts import read_chunks
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def test_read_chunks_happy_path(seeded_project, capsys):
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "document"
    assert out["document"]["file_name"] == "alpha.pdf"
    assert out["total"] == 4
    assert len(out["chunks"]) == 4
    indexes = [c["chunk_index"] for c in out["chunks"]]
    assert indexes == [0, 1, 2, 3]
    assert out["chunks"][0]["section_heading"] == "Intro"


def test_read_chunks_pagination(seeded_project, capsys):
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--offset", "1", "--limit", "2",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 4
    assert [c["chunk_index"] for c in out["chunks"]] == [1, 2]


def test_read_chunks_unknown_document(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", "999",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "DOCUMENT_NOT_FOUND"


def _doc_chunk_ids(project: str, document_id: int) -> list[int]:
    conn = open_db(project)
    try:
        rows = conn.cursor().execute(
            "SELECT chunk_id FROM chunks "
            "WHERE source_kind = 'document' AND source_id = ? "
            "ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def test_read_chunks_by_id_returns_requested(seeded_project, capsys):
    chunk_ids = _doc_chunk_ids(seeded_project["project"], seeded_project["doc_a"])
    target = [chunk_ids[2], chunk_ids[0]]  # arbitrary order

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", ",".join(str(c) for c in target),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "chunks"
    assert out["requested"] == target
    assert out["missing"] == []
    # Returned in requested order, with source metadata populated.
    assert [c["chunk_id"] for c in out["chunks"]] == target
    for c in out["chunks"]:
        assert c["source_kind"] == "document"
        assert c["source_id"] == seeded_project["doc_a"]
        assert c["source_name"] == "alpha.pdf"
        assert "chunk_index" in c


def test_read_chunks_by_id_reports_missing(seeded_project, capsys):
    chunk_ids = _doc_chunk_ids(seeded_project["project"], seeded_project["doc_a"])
    target = [chunk_ids[0], 999999]

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", ",".join(str(c) for c in target),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == [999999]
    assert [c["chunk_id"] for c in out["chunks"]] == [chunk_ids[0]]


def test_read_chunks_requires_document_or_chunks(seeded_project, capsys):
    with pytest.raises(SystemExit):
        read_chunks.main(["--project", seeded_project["project"]])


def test_read_chunks_document_and_chunks_mutually_exclusive(seeded_project, capsys):
    with pytest.raises(SystemExit):
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
            "--chunks", "1,2,3",
        ])
