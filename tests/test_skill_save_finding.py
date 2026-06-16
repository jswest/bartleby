"""Smoke test for skill/scripts/save_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import save_finding
from bartleby.db.connection import open_db
from tests._skill_fixtures import (  # noqa: F401
    assert_chunk_tables_consistent,
    mock_embed,
    project_env,
    seeded_project,
    unprefix,
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
        f"# A finding\n\nFirst claim[^chunk:{cited_ids[0]}].\n\n"
        f"Second claim[^chunk:{cited_ids[1]}]."
    )
    body_file.write_text(body_text, encoding="utf-8")

    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "PM25 equity",
        "--description", "Who bears the brunt of PM2.5 monitoring gaps.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    finding_id = unprefix(out["finding_id"])
    assert finding_id >= 1
    assert len(out["citations"]) == 2
    assert len(out["chunk_ids"]) == 1
    assert out["session_name"]
    assert out["body"] == body_text
    for c in out["citations"]:
        assert c["source_kind"] == "document"
        assert c["source_name"] == "alpha.pdf"
        assert c["file_name"] == "alpha.pdf"
        assert c["page_number"] is None
    assert [c["chunk_id"] for c in out["citations"]] == [
        f"chunk:{cid}" for cid in cited_ids
    ]

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        title, description, body = cur.execute(
            "SELECT title, description, body FROM findings WHERE finding_id = ?",
            (finding_id,),
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
            (finding_id,),
        ).fetchone()[0]
        assert n_citations == 2
    finally:
        conn.close()


def test_save_finding_memory_off_session_can_write(seeded_project, tmp_path, capsys):
    """The write half of the memory wall stays open: a ``memory_enabled=0``
    session can still author its own finding, and the new row is attributed to
    that session. Pins the "save stays open" enforcement shape
    (``ARCHITECTURE.md`` "Memory-off enforcement") so a blanket
    block-on-memory-off refactor can't pass the suite while breaking the eval
    workflow."""
    from bartleby.session import start_session

    project = seeded_project["project"]
    info = start_session(project, memory_enabled=False)

    conn = open_db(project)
    try:
        cited_id = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    body_file = tmp_path / "finding.md"
    body_file.write_text(f"A memory-off claim[^chunk:{cited_id}].", encoding="utf-8")
    save_finding.main([
        "--project", project,
        "--title", "memory-off write",
        "--description", "Writing under a memory-off session.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)

    # Success: the finding was written with its citation intact ...
    finding_id = unprefix(out["finding_id"])
    assert finding_id >= 1
    assert [c["chunk_id"] for c in out["citations"]] == [f"chunk:{cited_id}"]
    # ... and attributed to the memory-off session that authored it.
    assert out["session_id"] == info["session_id"]
    assert out["session_name"] == info["name"]

    conn = open_db(project)
    try:
        row_session_id = conn.cursor().execute(
            "SELECT session_id FROM findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()[0]
        assert row_session_id == info["session_id"]
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
    body_file.write_text(
        f"X[^chunk:{a}] Y[^chunk:{b}] Z[^chunk:{a}].", encoding="utf-8"
    )
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "dedup",
        "--description", "x",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert [c["chunk_id"] for c in out["citations"]] == [f"chunk:{a}", f"chunk:{b}"]


def test_save_finding_citations_include_page_number_when_available(
    seeded_project, tmp_path, capsys
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
    body_file.write_text(f"Cited from page 7[^chunk:{new_chunk_id}].", encoding="utf-8")
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
        "Real claim[^chunk:1] and typoed claim[3] in the same body.",
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
    body_file.write_text("body[^chunk:999999].", encoding="utf-8")
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


def test_finding_scripts_surface_session_provenance(seeded_project, tmp_path, capsys):
    """save/read/list all report the authoring session's model + harness."""
    from bartleby.session import set_session_provenance
    from bartleby.skill_scripts import list_findings, read_finding

    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cid = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    body_file = tmp_path / "f.md"
    body_file.write_text(f"Claim[^chunk:{cid}].", encoding="utf-8")
    save_finding.main([
        "--project", project, "--title", "t", "--description", "d",
        "--body-file", str(body_file),
    ])
    saved = json.loads(capsys.readouterr().out)
    finding_id = saved["finding_id"]  # already type-tagged, e.g. "finding:1"
    # save_finding always reports the keys (value may be env-derived at create time).
    assert "model" in saved and "harness" in saved

    # Stamp the authoring (active) session, then confirm read/list reflect it.
    set_session_provenance(project, model="qwen3.6:35b-mlx", harness="ollama-cli")

    read_finding.main(["--project", project, "--finding-id", finding_id])
    read_out = json.loads(capsys.readouterr().out)
    assert read_out["model"] == "qwen3.6:35b-mlx"
    assert read_out["harness"] == "ollama-cli"

    list_findings.main(["--project", project])
    list_out = json.loads(capsys.readouterr().out)
    row = next(f for f in list_out["findings"] if f["finding_id"] == finding_id)
    assert row["model"] == "qwen3.6:35b-mlx"
    assert row["harness"] == "ollama-cli"


def test_save_finding_failure_mid_write_leaves_no_trace(
    seeded_project, tmp_path, capsys, monkeypatch
):
    """A failure during chunk insertion (after the findings row is inserted)
    rolls back the whole call: no findings row, no finding chunks in any of the
    three chunk tables, no citation rows (issue #340). The runner seam wraps
    ``work()`` in one transaction, so the earlier INSERT INTO findings unwinds
    with the failed chunk write.

    The failure is injected at ``chunks._pack_embedding`` — i.e. *during* the
    chunk insert, after the findings row already landed — so this is a genuine
    mid-write rollback, not a pre-write validation bounce."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cited_id = cur.execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
        findings_before = cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
        finding_chunks_before = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding'"
        ).fetchone()[0]
        citations_before = cur.execute(
            "SELECT COUNT(*) FROM finding_citations"
        ).fetchone()[0]
    finally:
        conn.close()

    def _boom(embedding):
        raise RuntimeError("injected chunk-write failure")

    monkeypatch.setattr("bartleby.db.chunks._pack_embedding", _boom)

    body_file = tmp_path / "f.md"
    body_file.write_text(f"A claim[^chunk:{cited_id}].", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "doomed",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Every affected table is exactly as before — no orphaned finding row,
        # no chunks in any of the three chunk tables, no citations.
        assert cur.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == \
            findings_before
        assert cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding'"
        ).fetchone()[0] == finding_chunks_before
        assert cur.execute(
            "SELECT COUNT(*) FROM finding_citations"
        ).fetchone()[0] == citations_before
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


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


def test_save_finding_response_echoes_title_and_description(
    seeded_project, tmp_path, capsys
):
    """save_finding echoes title and description in the response so a
    shell-mangled title is visible without a follow-up read_finding."""
    conn = open_db(seeded_project["project"])
    try:
        cited_id = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    body_file = tmp_path / "f.md"
    body_file.write_text(f"Claim[^chunk:{cited_id}].", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "Echo test title",
        "--description", "Echo test description.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["title"] == "Echo test title"
    assert out["description"] == "Echo test description."


def test_save_finding_title_file_reads_verbatim(seeded_project, tmp_path, capsys):
    """--title-file reads the title verbatim from a file — dollar signs, backticks,
    and parens must survive with no shell expansion."""
    conn = open_db(seeded_project["project"])
    try:
        cited_id = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    tricky_title = "Spending ~$8M/quarter (in-house `estimate`)"
    title_file = tmp_path / "title.txt"
    title_file.write_text(tricky_title, encoding="utf-8")

    body_file = tmp_path / "f.md"
    body_file.write_text(f"Claim[^chunk:{cited_id}].", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title-file", str(title_file),
        "--description", "desc",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["title"] == tricky_title

    # Confirm it landed in the DB verbatim too.
    conn = open_db(seeded_project["project"])
    try:
        db_title = conn.cursor().execute(
            "SELECT title FROM findings WHERE finding_id = ?",
            (unprefix(out["finding_id"]),),
        ).fetchone()[0]
    finally:
        conn.close()
    assert db_title == tricky_title


def test_save_finding_description_file_reads_verbatim(
    seeded_project, tmp_path, capsys
):
    """--description-file reads the description verbatim, including shell-unsafe chars."""
    conn = open_db(seeded_project["project"])
    try:
        cited_id = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    tricky_desc = "Revenue is $TOTAL (see `quarterly_report.pdf`)"
    desc_file = tmp_path / "desc.txt"
    desc_file.write_text(tricky_desc, encoding="utf-8")

    body_file = tmp_path / "f.md"
    body_file.write_text(f"Claim[^chunk:{cited_id}].", encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "t",
        "--description-file", str(desc_file),
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["description"] == tricky_desc
