"""Smoke test for skill/scripts/delete_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import delete_finding, save_finding
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr(
        "bartleby.skill_scripts._common.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.skill_scripts._common.chunk_markdown_string",
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
        "--finding-id", str(finding_id),
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
        # The finding's body chunks are gone from chunks + the fts/vec mirrors.
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding' "
            "AND source_id = ?",
            (finding_id,),
        ).fetchone()[0] == 0
        for cid in body_chunk_ids:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE rowid = ?", (cid,)
            ).fetchone()[0] == 0
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,)
            ).fetchone()[0] == 0
        # The cited *document* chunks (evidence) are untouched.
        for cid in (a, b):
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_id = ?", (cid,)
            ).fetchone()[0] == 1
    finally:
        conn.close()


def test_delete_finding_unknown_id(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        delete_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", "99999",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "FINDING_NOT_FOUND"
