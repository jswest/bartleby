"""Smoke test for skill/scripts/edit_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import edit_finding, save_finding
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    # Both scripts share `embed_texts` / `chunk_markdown_string` via _common;
    # patch there so the test never reaches BAAI.
    monkeypatch.setattr(
        "bartleby.skill_scripts._common.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.skill_scripts._common.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def _seed_finding(seeded_project, tmp_path, capsys, *, body_suffix: str = "") -> dict:
    """Save a baseline finding and return the parsed save_finding response."""
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

    body_file = tmp_path / "initial.md"
    body_file.write_text(
        f"# Original\n\nClaim one[^{a}]. Claim two[^{b}].{body_suffix}",
        encoding="utf-8",
    )
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "Original title",
        "--description", "Original description.",
        "--body-file", str(body_file),
    ])
    saved = json.loads(capsys.readouterr().out)
    saved["_chunks"] = (a, b)
    return saved


def test_edit_finding_body_rebuilds_citations_and_chunks(
    seeded_project, tmp_path, capsys
):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]
    a, b = saved["_chunks"]

    # Fetch a third chunk to verify swapping citations works.
    conn = open_db(seeded_project["project"])
    try:
        c = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? AND chunk_index = 2",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    new_body = f"# Fixed\n\nOnly claim now[^{c}]."
    new_body_file = tmp_path / "edited.md"
    new_body_file.write_text(new_body, encoding="utf-8")

    edit_finding.main([
        "--project", seeded_project["project"],
        "--finding", str(finding_id),
        "--body-file", str(new_body_file),
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["finding_id"] == finding_id
    assert out["body"] == new_body
    assert [cite["chunk_id"] for cite in out["citations"]] == [c]
    assert out["session_name"]  # owning session round-tripped

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        title, description, body = cur.execute(
            "SELECT title, description, body FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        assert title == "Original title"
        assert description == "Original description."
        assert body == new_body

        citation_rows = list(cur.execute(
            "SELECT chunk_id FROM finding_citations WHERE finding_id = ? "
            "ORDER BY chunk_id",
            (finding_id,),
        ))
        assert [r[0] for r in citation_rows] == [c]

        # The finding's current chunks reflect the new body, not the old one.
        # (SQLite reuses chunk_id values after delete, so we check by text,
        # not by id disjointness.)
        finding_chunk_texts = [
            row[0] for row in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (finding_id,),
            )
        ]
        assert finding_chunk_texts == [new_body]
        # The chunks_fts mirror is in sync with the new body too.
        finding_chunk_ids = [
            row[0] for row in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ?",
                (finding_id,),
            )
        ]
        for cid in finding_chunk_ids:
            fts_text = cur.execute(
                "SELECT text FROM chunks_fts WHERE rowid = ?", (cid,)
            ).fetchone()
            assert fts_text == (new_body,)
    finally:
        conn.close()


def test_edit_finding_title_only_leaves_body_and_citations_intact(
    seeded_project, tmp_path, capsys
):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        original_chunk_ids = [
            row[0] for row in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (finding_id,),
            )
        ]
        original_citation_ids = [
            row[0] for row in cur.execute(
                "SELECT chunk_id FROM finding_citations WHERE finding_id = ? "
                "ORDER BY chunk_id",
                (finding_id,),
            )
        ]
    finally:
        conn.close()

    edit_finding.main([
        "--project", seeded_project["project"],
        "--finding", str(finding_id),
        "--title", "Renamed title",
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["body"] == saved["body"]
    assert out["chunk_ids"] == original_chunk_ids
    assert [c["chunk_id"] for c in out["citations"]] == original_citation_ids

    conn = open_db(seeded_project["project"])
    try:
        title, description = conn.cursor().execute(
            "SELECT title, description FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        assert title == "Renamed title"
        assert description == "Original description."  # untouched
    finally:
        conn.close()


def test_edit_finding_requires_at_least_one_field(seeded_project, tmp_path, capsys):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding", str(saved["finding_id"]),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NOTHING_TO_UPDATE"


def test_edit_finding_unknown_id(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding", "99999",
            "--title", "x",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "FINDING_NOT_FOUND"


def test_edit_finding_rejects_body_without_citations(
    seeded_project, tmp_path, capsys
):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    new_body_file = tmp_path / "no-cites.md"
    new_body_file.write_text("Just prose, no citations.", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding", str(saved["finding_id"]),
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NO_INLINE_CITATIONS"


def test_edit_finding_rejects_malformed_citation(
    seeded_project, tmp_path, capsys
):
    """Bare ``[N]`` markers in a replacement body are refused before the
    new body lands in the DB."""
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    new_body_file = tmp_path / "typo.md"
    new_body_file.write_text(
        "Real[^1] but typoed[3] and again[42].",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding", str(saved["finding_id"]),
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MALFORMED_CITATION"
    assert out["malformed_markers"] == ["[3]", "[42]"]


def test_edit_finding_rejects_unknown_chunk_marker(
    seeded_project, tmp_path, capsys
):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    new_body_file = tmp_path / "bad-cite.md"
    new_body_file.write_text("Garbage[^999999].", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding", str(saved["finding_id"]),
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_CITATIONS"
    assert out["unknown_chunk_ids"] == [999999]


def test_edit_finding_rejects_empty_title(seeded_project, tmp_path, capsys):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding", str(saved["finding_id"]),
            "--title", "   ",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_TITLE"
