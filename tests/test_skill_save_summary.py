"""Smoke test for skill/scripts/save_summary.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import save_summary
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr(
        "bartleby.skill_scripts.save_summary.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    # Force single-chunk output so chunk_count assertions stay stable.
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.skill_scripts.save_summary.chunk_markdown_string",
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
    # alpha already has an ingest-time summary; replace it.
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
        rows = conn.cursor().execute(
            "SELECT title, text FROM summaries WHERE document_id = ?",
            (seeded_project["doc_a"],),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == ("Alpha v2", "Replacement summary for alpha.")
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
