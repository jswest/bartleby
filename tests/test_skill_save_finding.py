"""Smoke test for skill/scripts/save_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import save_finding
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr(
        "bartleby.skill_scripts.save_finding.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.skill_scripts.save_finding.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def test_save_finding_with_citations(seeded_project, tmp_path, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cited = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 2",
            (seeded_project["doc_a"],),
        ).fetchall()
        cited_ids = [r[0] for r in cited]
    finally:
        conn.close()

    body_file = tmp_path / "finding.md"
    body_file.write_text("# A finding\n\nThis is the body.", encoding="utf-8")

    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "PM25 equity",
        "--body-file", str(body_file),
        "--citations", ",".join(str(c) for c in cited_ids),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["finding_id"] >= 1
    assert out["citation_count"] == 2
    assert len(out["chunk_ids"]) == 1
    assert out["session_name"]

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        title, body = cur.execute(
            "SELECT title, body FROM findings WHERE finding_id = ?",
            (out["finding_id"],),
        ).fetchone()
        assert title == "PM25 equity"
        assert "This is the body" in body

        n_citations = cur.execute(
            "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
            (out["finding_id"],),
        ).fetchone()[0]
        assert n_citations == 2
    finally:
        conn.close()


def test_save_finding_without_citations(seeded_project, tmp_path, capsys):
    body_file = tmp_path / "f.md"
    body_file.write_text("Just a note.", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "no-citations",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["citation_count"] == 0


def test_save_finding_unknown_citation_chunk(seeded_project, tmp_path, capsys):
    body_file = tmp_path / "f.md"
    body_file.write_text("body", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "bad",
            "--body-file", str(body_file),
            "--citations", "999999",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_CITATIONS"


def test_save_finding_missing_body_file(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "x",
            "--body-file", "/tmp/does-not-exist-zzz.md",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "BODY_FILE_NOT_FOUND"
