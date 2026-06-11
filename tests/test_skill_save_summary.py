"""Smoke test for skill/scripts/save_summary.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import save_summary
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from tests._skill_fixtures import (  # noqa: F401
    assert_chunk_tables_consistent,
    project_env,
    seeded_project,
)


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    # save_summary embeds via _common.embed_body_chunks; patch the names where
    # that helper looks them up.
    monkeypatch.setattr(
        "bartleby.ingest.embed.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    # Force single-chunk output so chunk_count assertions stay stable.
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.ingest.chunk.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def test_save_summary_creates_new_summary(seeded_project, capsys):
    save_summary.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_b"]),
        "--title", "Beta paper",
        "--description", "An agent's take on beta's main argument.",
        "--text", "Agent-authored summary of beta.",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["document_id"] == seeded_project["doc_b"]
    assert out["summary_id"] >= 1
    assert len(out["chunk_ids"]) == 1

    conn = open_db(seeded_project["project"])
    try:
        row = conn.cursor().execute(
            "SELECT title, description, text FROM summaries WHERE document_id = ?",
            (seeded_project["doc_b"],),
        ).fetchone()
        assert row == (
            "Beta paper",
            "An agent's take on beta's main argument.",
            "Agent-authored summary of beta.",
        )
    finally:
        conn.close()


def test_save_summary_replaces_existing(seeded_project, capsys):
    # alpha already has an ingest-time summary, seeded WITH chunks; replace it
    # and verify the prior summary's chunks are purged from all three tables.
    # SQLite reuses both summary_id and chunk_id rowids, so the prior summary
    # is identified by its distinctive chunk TEXT, not by id (an id check would
    # pass vacuously when the replacement re-owns the freed rowids).
    prior_chunk_ids = seeded_project["summary_a_chunk_ids"]
    assert prior_chunk_ids  # sanity: the prior summary actually had chunks

    conn = open_db(seeded_project["project"])
    try:
        prior_chunk_texts = [
            r[0] for r in conn.cursor().execute(
                "SELECT text FROM chunks WHERE source_kind='summary' "
                "AND chunk_id IN ({})".format(
                    ",".join("?" * len(prior_chunk_ids))
                ),
                tuple(prior_chunk_ids),
            )
        ]
    finally:
        conn.close()
    assert len(prior_chunk_texts) == len(prior_chunk_ids)

    save_summary.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--title", "Alpha v2",
        "--description", "Replacement description for alpha.",
        "--text", "Replacement summary for alpha.",
    ])
    capsys.readouterr()  # discard

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT title, text FROM summaries WHERE document_id = ?",
            (seeded_project["doc_a"],),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == ("Alpha v2", "Replacement summary for alpha.")

        # The prior summary's chunk texts are gone from chunks entirely: the
        # replace deleted the old chunks before inserting the new body. (If the
        # old chunks lingered, these distinctive texts would still be present.)
        for text in prior_chunk_texts:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE text = ?", (text,),
            ).fetchone()[0] == 0

        # The surviving summary's chunks reflect the NEW body, not the old one.
        new_summary_chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT c.text FROM chunks c "
                "JOIN summaries s ON s.summary_id = c.source_id "
                "WHERE c.source_kind='summary' AND s.document_id = ?",
                (seeded_project["doc_a"],),
            )
        ]
        assert new_summary_chunk_texts == ["Replacement summary for alpha."]

        # And all three tables agree: the FTS integrity check would fail if a
        # stale prior-chunk entry survived in the index, and the vec rowid set
        # must equal the chunks rowid set.
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


def test_save_summary_accepts_iso_authored_date(seeded_project, capsys):
    save_summary.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_b"]),
        "--title", "Beta",
        "--description", "Dated.",
        "--text", "body",
        "--authored-date", "2024-09-12",
    ])
    capsys.readouterr()
    conn = open_db(seeded_project["project"])
    try:
        row = conn.cursor().execute(
            "SELECT authored_date FROM summaries WHERE document_id = ?",
            (seeded_project["doc_b"],),
        ).fetchone()
        assert row[0] == "2024-09-12"
    finally:
        conn.close()


def test_save_summary_drops_malformed_authored_date(seeded_project, capsys):
    save_summary.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_b"]),
        "--title", "Beta",
        "--description", "Dated.",
        "--text", "body",
        "--authored-date", "Q3 2024",
    ])
    capsys.readouterr()
    conn = open_db(seeded_project["project"])
    try:
        row = conn.cursor().execute(
            "SELECT authored_date FROM summaries WHERE document_id = ?",
            (seeded_project["doc_b"],),
        ).fetchone()
        assert row[0] is None
    finally:
        conn.close()


def test_save_summary_failed_replace_preserves_prior_summary_and_chunks(
    seeded_project, capsys, monkeypatch
):
    """A failure mid-replace leaves the prior summary AND its chunks fully
    intact (issue #340). The replace path deletes the prior summary and its
    chunks, inserts the new summary, then inserts the new chunks — under the
    runner's transaction wrap, a failure during that last chunk insert rolls
    the *whole* sequence back, so the prior summary (and its inferred
    authored_date) is not destroyed and ``summaries.document_id`` (UNIQUE) is
    not orphaned.

    The failure is injected at ``chunks._pack_embedding`` — during the new chunk
    insert, *after* the prior summary + chunks were deleted and the new summary
    row inserted — so it is a genuine mid-write rollback, not a pre-write bounce.
    (Embedding itself is hoisted ahead of the first write, so a bad-embedding
    vector would fail before any delete; breaking the chunk insert directly is
    what reaches the post-delete state this test pins.)"""
    doc_a = seeded_project["doc_a"]
    prior_chunk_ids = seeded_project["summary_a_chunk_ids"]
    assert prior_chunk_ids  # the prior summary has chunks to preserve

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        before_summary = cur.execute(
            "SELECT title, description, text FROM summaries WHERE document_id = ?",
            (doc_a,),
        ).fetchone()
        before_chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT text FROM chunks WHERE source_kind='summary' "
                "AND chunk_id IN ({}) ORDER BY chunk_index".format(
                    ",".join("?" * len(prior_chunk_ids))
                ),
                tuple(prior_chunk_ids),
            )
        ]
    finally:
        conn.close()
    assert before_summary is not None
    assert len(before_chunk_texts) == len(prior_chunk_ids)

    def _boom(embedding):
        raise RuntimeError("injected chunk-write failure")

    monkeypatch.setattr("bartleby.db.chunks._pack_embedding", _boom)

    with pytest.raises(SystemExit) as exc:
        save_summary.main([
            "--project", seeded_project["project"],
            "--document", str(doc_a),
            "--title", "Doomed v2",
            "--description", "Replacement that must roll back.",
            "--text", "Replacement summary that never lands.",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # The prior summary survives, unchanged — exactly one row, original text.
        rows = cur.execute(
            "SELECT title, description, text FROM summaries WHERE document_id = ?",
            (doc_a,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == before_summary
        # The prior summary's chunks survive in all three tables, byte-for-byte.
        chunk_texts = [
            r[0] for r in cur.execute(
                "SELECT c.text FROM chunks c "
                "JOIN summaries s ON s.summary_id = c.source_id "
                "WHERE c.source_kind='summary' AND s.document_id = ? "
                "ORDER BY c.chunk_index",
                (doc_a,),
            )
        ]
        assert chunk_texts == before_chunk_texts
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


def test_save_summary_unknown_document(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        save_summary.main([
            "--project", seeded_project["project"],
            "--document", "999",
            "--title", "x",
            "--description", "x",
            "--text", "irrelevant",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "DOCUMENT_NOT_FOUND"
