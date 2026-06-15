"""Smoke test for skill/scripts/edit_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import edit_finding
from bartleby.db.connection import open_db
from tests._skill_fixtures import (  # noqa: F401
    assert_chunk_tables_consistent,
    mock_embed,
    project_env,
    seed_finding_via_main,
    seeded_project,
)


def _seed_finding(seeded_project, tmp_path, capsys, *, body_suffix: str = "") -> dict:
    """Seed the baseline finding these edit tests assert against ("Original …")."""
    return seed_finding_via_main(
        seeded_project, tmp_path, capsys,
        title="Original title", description="Original description.",
        body_suffix=body_suffix,
    )


def test_edit_finding_body_rebuilds_citations_and_chunks(
    seeded_project, tmp_path, capsys
):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]
    a, b = saved["_chunks"]

    # Fetch a third chunk to verify swapping citations works, and capture the
    # finding's current body-chunk ids before the edit replaces them.
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        c = cur.execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? AND chunk_index = 2",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
        old_finding_chunk_ids = {
            row[0] for row in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ?",
                (finding_id,),
            )
        }
    finally:
        conn.close()
    assert old_finding_chunk_ids  # sanity: the finding had body chunks

    new_body = f"# Fixed\n\nOnly claim now[^{c}]."
    new_body_file = tmp_path / "edited.md"
    new_body_file.write_text(new_body, encoding="utf-8")

    edit_finding.main([
        "--project", seeded_project["project"],
        "--finding-id", str(finding_id),
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
        finding_chunk_ids = [
            row[0] for row in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ?",
                (finding_id,),
            )
        ]
        # The chunks_fts mirror is in sync with the new body too; that is
        # asserted by the triple-table sync guard below (a per-rowid read of
        # the external-content chunks_fts goes THROUGH chunks and is vacuous).
        # The chunks_vec mirror tracks the rebuild: the new body chunks are
        # present in vec, and any old chunk id NOT reused by the new body is
        # gone. (SQLite reuses chunk_id values, so disjointness can't be
        # assumed — only the ids that fell out of the finding must be absent.)
        new_finding_chunk_ids = set(finding_chunk_ids)
        for cid in new_finding_chunk_ids:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,),
            ).fetchone()[0] == 1
        for cid in old_finding_chunk_ids - new_finding_chunk_ids:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,),
            ).fetchone()[0] == 0

        assert_chunk_tables_consistent(conn)
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
        "--finding-id", str(finding_id),
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
            "--finding-id", str(saved["finding_id"]),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NOTHING_TO_UPDATE"


def test_edit_finding_unknown_id(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", "99999",
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
            "--finding-id", str(saved["finding_id"]),
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
            "--finding-id", str(saved["finding_id"]),
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
            "--finding-id", str(saved["finding_id"]),
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_CITATIONS"
    assert out["unknown_chunk_ids"] == [999999]


def test_edit_finding_rejects_citation_to_own_chunk(
    seeded_project, tmp_path, capsys
):
    """A replacement body citing the finding's *own* pre-edit body chunk is
    refused upfront with CITES_OWN_CHUNKS naming the offending id, instead of
    crashing with a bare INTERNAL_ERROR (the chunk would be deleted before its
    citation row is replaced). The finding is left untouched."""
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]
    a, _ = saved["_chunks"]

    conn = open_db(seeded_project["project"])
    try:
        own_chunk = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (finding_id,),
        ).fetchone()[0]
    finally:
        conn.close()

    new_body_file = tmp_path / "self-cite.md"
    new_body_file.write_text(
        f"# Reworked\n\nValid doc cite[^{a}] plus my own chunk[^{own_chunk}].",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", str(finding_id),
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "CITES_OWN_CHUNKS"
    assert out["offending_chunk_ids"] == [own_chunk]

    # The edit aborted before any write: the finding's title and body survive.
    conn = open_db(seeded_project["project"])
    try:
        title, body = conn.cursor().execute(
            "SELECT title, body FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        assert title == "Original title"
        assert body == saved["body"]
    finally:
        conn.close()


def test_edit_finding_memory_off_other_session(seeded_project, tmp_path, capsys):
    """A memory-off session cannot edit (and thereby read back) a finding
    authored by another session — the response echoes the body, so an ungated
    --title-only edit would be a read-by-write bypass of the memory wall."""
    from bartleby.session import start_session

    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]

    # The seed finding belongs to the (memory-on) default session; open a
    # fresh memory-off session as the would-be attacker.
    start_session(seeded_project["project"], memory_enabled=False)

    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", str(finding_id),
            "--title", "Hijacked title",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"
    # No body or citation content leaks back through the error.
    assert "body" not in out
    assert "citations" not in out

    # And the foreign finding was not mutated.
    conn = open_db(seeded_project["project"])
    try:
        title = conn.cursor().execute(
            "SELECT title FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()[0]
        assert title == "Original title"
    finally:
        conn.close()


def test_edit_finding_memory_off_own_session(seeded_project, tmp_path, capsys):
    """A memory-off session can still edit findings it authored itself."""
    from bartleby.session import start_session

    start_session(seeded_project["project"], memory_enabled=False)
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]

    edit_finding.main([
        "--project", seeded_project["project"],
        "--finding-id", str(finding_id),
        "--title", "Self-renamed",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["finding_id"] == finding_id
    assert out["body"] == saved["body"]  # own body still echoes back

    conn = open_db(seeded_project["project"])
    try:
        title = conn.cursor().execute(
            "SELECT title FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()[0]
        assert title == "Self-renamed"
    finally:
        conn.close()


def test_edit_finding_failure_mid_write_leaves_finding_intact(
    seeded_project, tmp_path, capsys, monkeypatch
):
    """A failure during the body re-chunk (after the findings UPDATE) rolls back
    the whole edit: the prior title, body, chunks, and citations all survive
    intact (issue #340). Without the transaction wrap, the UPDATE and the
    ``delete_chunks_for`` inside ``write_finding_chunks`` would have committed
    independently, leaving the new body saved with zero chunks and stale
    citations. The failure is injected at ``chunks._pack_embedding`` — during
    the chunk insert, after the UPDATE — so it's a true mid-write rollback."""
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    finding_id = saved["finding_id"]
    a, b = saved["_chunks"]

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        before_title, before_body = cur.execute(
            "SELECT title, body FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        before_chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (finding_id,),
            )
        ]
        before_citations = sorted(
            r[0] for r in cur.execute(
                "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
                (finding_id,),
            )
        )
    finally:
        conn.close()
    assert before_chunk_texts  # sanity: the finding had body chunks

    def _boom(embedding):
        raise RuntimeError("injected chunk-write failure")

    monkeypatch.setattr("bartleby.db.chunks._pack_embedding", _boom)

    # Edit the body (and title) — the rebuild must fail and roll everything back.
    new_body_file = tmp_path / "doomed.md"
    new_body_file.write_text(f"# Doomed\n\nNew claim[^{a}].", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", str(finding_id),
            "--title", "Doomed title",
            "--body-file", str(new_body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Title and body unchanged — the UPDATE rolled back with the failed
        # chunk write.
        title, body = cur.execute(
            "SELECT title, body FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        assert title == before_title
        assert body == before_body
        # The original body chunks survive, byte-for-byte.
        chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (finding_id,),
            )
        ]
        assert chunk_texts == before_chunk_texts
        # And the original citations survive.
        citations = sorted(
            r[0] for r in cur.execute(
                "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
                (finding_id,),
            )
        )
        assert citations == before_citations
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


def test_edit_finding_rejects_empty_title(seeded_project, tmp_path, capsys):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", str(saved["finding_id"]),
            "--title", "   ",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_TITLE"


def test_edit_finding_rejects_empty_description(seeded_project, tmp_path, capsys):
    saved = _seed_finding(seeded_project, tmp_path, capsys)
    with pytest.raises(SystemExit) as exc:
        edit_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", str(saved["finding_id"]),
            "--description", "   ",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_DESCRIPTION"
