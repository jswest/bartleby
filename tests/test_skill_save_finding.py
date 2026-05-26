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
        "bartleby.skill_scripts._common.embed_texts",
        lambda texts: [[0.01 * i for _ in range(EMBEDDING_DIM)] for i in range(len(texts))],
    )
    from bartleby.ingest.chunk import ChunkRow
    monkeypatch.setattr(
        "bartleby.skill_scripts._common.chunk_markdown_string",
        lambda md: [ChunkRow(text=md, section_heading=None, content_type=None)],
    )


def test_save_finding_with_inline_citations(seeded_project, tmp_path, capsys):
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
    body_text = (
        f"# A finding\n\nFirst claim[^{cited_ids[0]}].\n\n"
        f"Second claim[^{cited_ids[1]}]."
    )
    body_file.write_text(body_text, encoding="utf-8")

    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "PM25 equity",
        "--description", "Who bears the brunt of PM2.5 monitoring gaps.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["finding_id"] >= 1
    assert len(out["citations"]) == 2
    assert len(out["chunk_ids"]) == 1
    assert out["session_name"]
    assert out["body"] == body_text
    for c in out["citations"]:
        assert c["source_kind"] == "document"
        assert c["source_name"] == "alpha.pdf"
        assert c["file_name"] == "alpha.pdf"
        assert c["page_number"] is None
    assert [c["chunk_id"] for c in out["citations"]] == cited_ids

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        title, description, body = cur.execute(
            "SELECT title, description, body FROM findings WHERE finding_id = ?",
            (out["finding_id"],),
        ).fetchone()
        assert title == "PM25 equity"
        assert description == "Who bears the brunt of PM2.5 monitoring gaps."
        # The returned body must match the DB byte-for-byte — that's the
        # single-source-of-truth contract the agent relies on to echo it back
        # to the user without drift.
        assert body == body_text
        assert out["body"] == body

        n_citations = cur.execute(
            "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
            (out["finding_id"],),
        ).fetchone()[0]
        assert n_citations == 2
    finally:
        conn.close()


def test_save_finding_dedupes_repeated_markers(seeded_project, tmp_path, capsys):
    """Same chunk cited twice → one finding_citations row, preserving order."""
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
    body_file.write_text(f"X[^{a}] Y[^{b}] Z[^{a}].", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "dedup",
        "--description", "x",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert [c["chunk_id"] for c in out["citations"]] == [a, b]


def test_save_finding_citations_include_page_number_when_available(
    seeded_project, tmp_path, capsys, monkeypatch
):
    """A first-class page_number on the chunk surfaces in citations."""
    from bartleby.db.connection import open_db
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM

    conn = open_db(seeded_project["project"])
    try:
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_document_chunks(conn, seeded_project["doc_a"], [
            ChunkInput(
                text="alpha chunk four on page 7",
                embedding=emb, chunk_index=4,
                section_heading=None, page_number=7, content_type="text",
            ),
        ])
        new_chunk_id = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id=? AND chunk_index=4",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    body_file = tmp_path / "f.md"
    body_file.write_text(f"Cited from page 7[^{new_chunk_id}].", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "page test",
        "--description", "Testing page number propagation.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    [cite] = out["citations"]
    assert cite["file_name"] == "alpha.pdf"
    assert cite["page_number"] == 7


def test_save_finding_no_inline_citations(seeded_project, tmp_path, capsys):
    """Bodies without [^N] markers are rejected — citations are mandatory."""
    body_file = tmp_path / "f.md"
    body_file.write_text("Just prose without any citations.", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "no cites",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NO_INLINE_CITATIONS"


def test_save_finding_rejects_malformed_citation(seeded_project, tmp_path, capsys):
    """Bare ``[N]`` markers (missing the ``^``) are rejected loudly so the
    agent fixes the typo instead of shipping silent phantom citations."""
    body_file = tmp_path / "f.md"
    body_file.write_text(
        "Real claim[^1] and typoed claim[3] in the same body.",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "typo",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MALFORMED_CITATION"
    assert out["malformed_markers"] == ["[3]"]


def test_save_finding_unknown_inline_marker(seeded_project, tmp_path, capsys):
    body_file = tmp_path / "f.md"
    body_file.write_text("body[^999999].", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "bad",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_CITATIONS"
    assert out["unknown_chunk_ids"] == [999999]


def test_save_finding_missing_body_file(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "x",
            "--description", "x",
            "--body-file", "/tmp/does-not-exist-zzz.md",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "BODY_FILE_NOT_FOUND"
