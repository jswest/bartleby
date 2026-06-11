"""Smoke test for skill/scripts/merge_findings.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import merge_findings, save_finding
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


def _doc_chunk_ids(project, doc_id) -> list[int]:
    conn = open_db(project)
    try:
        return [
            r[0] for r in conn.cursor().execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='document' "
                "AND source_id = ? ORDER BY chunk_index",
                (doc_id,),
            )
        ]
    finally:
        conn.close()


def _finding_chunk_ids(project, finding_id) -> list[int]:
    conn = open_db(project)
    try:
        return [
            r[0] for r in conn.cursor().execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ? ORDER BY chunk_index",
                (finding_id,),
            )
        ]
    finally:
        conn.close()


def _save(project, tmp_path, capsys, *, name, title, cite) -> int:
    body_file = tmp_path / f"{name}.md"
    body_file.write_text(f"# {title}\n\nClaim[^{cite}].", encoding="utf-8")
    save_finding.main([
        "--project", project,
        "--title", title,
        "--description", f"{title} description.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    return out["finding_id"]


def test_merge_folds_sources_into_target(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    chunks = _doc_chunk_ids(project, seeded_project["doc_a"])
    c0, c1, c2 = chunks[0], chunks[1], chunks[2]

    target = _save(project, tmp_path, capsys, name="t", title="Keep me", cite=c0)
    src1 = _save(project, tmp_path, capsys, name="s1", title="Dup one", cite=c1)
    src2 = _save(project, tmp_path, capsys, name="s2", title="Dup two", cite=c2)

    # Capture the source findings' body-chunk ids before the merge consumes
    # them, so we can assert they leave all three chunk tables.
    src_chunk_ids = _finding_chunk_ids(project, src1) + _finding_chunk_ids(project, src2)
    assert src_chunk_ids  # sanity: the sources had body chunks

    merged_body = f"# Consolidated\n\nAll together[^{c0}][^{c1}][^{c2}]."
    merged_file = tmp_path / "merged.md"
    merged_file.write_text(merged_body, encoding="utf-8")

    merge_findings.main([
        "--project", project,
        "--from", f"{src1},{src2}",
        "--into", str(target),
        "--body-file", str(merged_file),
        "--title", "SAP billing/arrears",
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["finding_id"] == target
    assert out["body"] == merged_body
    assert out["merged_from"] == [src1, src2]
    assert [c["chunk_id"] for c in out["citations"]] == [c0, c1, c2]
    # Output mirrors save/edit/read: full session provenance, not just the name.
    assert out["session_name"]
    assert "model" in out and "harness" in out

    conn = open_db(project)
    try:
        cur = conn.cursor()
        # Target survived with new title + body; description kept (omitted flag).
        title, description, body = cur.execute(
            "SELECT title, description, body FROM findings WHERE finding_id = ?",
            (target,),
        ).fetchone()
        assert title == "SAP billing/arrears"
        assert description == "Keep me description."
        assert body == merged_body
        assert [
            r[0] for r in cur.execute(
                "SELECT chunk_id FROM finding_citations WHERE finding_id = ? "
                "ORDER BY chunk_id", (target,),
            )
        ] == sorted([c0, c1, c2])

        # Sources are gone — rows, their body chunks, and their citations.
        for src in (src1, src2):
            assert cur.execute(
                "SELECT COUNT(*) FROM findings WHERE finding_id = ?", (src,),
            ).fetchone()[0] == 0
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE source_kind='finding' "
                "AND source_id = ?", (src,),
            ).fetchone()[0] == 0
            assert cur.execute(
                "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
                (src,),
            ).fetchone()[0] == 0

        # The source body chunks left BOTH mirrors, and the target's NEW body
        # chunks are present in both. The merged body re-chunks the target, so
        # its chunk ids are whatever the finding now owns.
        target_chunk_ids = [
            r[0] for r in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
                "AND source_id = ?", (target,),
            )
        ]
        assert target_chunk_ids  # sanity: the merged target has body chunks
        for cid in src_chunk_ids:
            if cid in target_chunk_ids:
                continue  # chunk_id reused by the rebuilt target body
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,),
            ).fetchone()[0] == 0
        for cid in target_chunk_ids:
            assert cur.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,),
            ).fetchone()[0] == 1

        # The FTS leg is covered by the triple-table sync guard below; a
        # per-rowid COUNT over the external-content chunks_fts reads THROUGH
        # chunks and is vacuous.
        assert_chunk_tables_consistent(conn)
    finally:
        conn.close()


def test_merge_surfaces_session_provenance(seeded_project, tmp_path, capsys):
    """The merged target reports its authoring session's model + harness,
    matching save/edit/read (issue #75)."""
    from bartleby.session import set_session_provenance

    project = seeded_project["project"]
    chunks = _doc_chunk_ids(project, seeded_project["doc_a"])
    c0, c1 = chunks[0], chunks[1]

    target = _save(project, tmp_path, capsys, name="t", title="Keep", cite=c0)
    src = _save(project, tmp_path, capsys, name="s", title="Dup", cite=c1)

    # Stamp the authoring (active) session so model/harness are non-NULL.
    set_session_provenance(project, model="qwen3.6:35b-mlx", harness="ollama-cli")

    merged_file = tmp_path / "merged.md"
    merged_file.write_text(f"# Consolidated\n\nTogether[^{c0}][^{c1}].",
                           encoding="utf-8")
    merge_findings.main([
        "--project", project,
        "--from", str(src),
        "--into", str(target),
        "--body-file", str(merged_file),
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["model"] == "qwen3.6:35b-mlx"
    assert out["harness"] == "ollama-cli"


def test_merge_missing_source_reports_ids(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    c0 = _doc_chunk_ids(project, seeded_project["doc_a"])[0]
    target = _save(project, tmp_path, capsys, name="t", title="Target", cite=c0)

    merged_file = tmp_path / "m.md"
    merged_file.write_text(f"Body[^{c0}].", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        merge_findings.main([
            "--project", project,
            "--from", "98765,98766",
            "--into", str(target),
            "--body-file", str(merged_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "FINDING_NOT_FOUND"
    assert out["missing_finding_ids"] == [98765, 98766]


def test_merge_target_in_sources_rejected(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    c0 = _doc_chunk_ids(project, seeded_project["doc_a"])[0]
    target = _save(project, tmp_path, capsys, name="t", title="Target", cite=c0)

    merged_file = tmp_path / "m.md"
    merged_file.write_text(f"Body[^{c0}].", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        merge_findings.main([
            "--project", project,
            "--from", str(target),
            "--into", str(target),
            "--body-file", str(merged_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TARGET_IN_SOURCES"


def test_merge_rejects_body_without_citations(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    chunks = _doc_chunk_ids(project, seeded_project["doc_a"])
    target = _save(project, tmp_path, capsys, name="t", title="Target", cite=chunks[0])
    src = _save(project, tmp_path, capsys, name="s", title="Source", cite=chunks[1])

    merged_file = tmp_path / "m.md"
    merged_file.write_text("Just prose, no citations.", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        merge_findings.main([
            "--project", project,
            "--from", str(src),
            "--into", str(target),
            "--body-file", str(merged_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NO_INLINE_CITATIONS"

    # The merge aborted before touching the DB: both findings still exist.
    conn = open_db(project)
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id IN (?, ?)",
            (target, src),
        ).fetchone()[0] == 2
    finally:
        conn.close()


@pytest.mark.parametrize("cite_role", ["source", "target"])
def test_merge_rejects_citation_to_involved_finding_chunk(
    seeded_project, tmp_path, capsys, cite_role
):
    """A merged body that cites a finding-kind chunk owned by a finding involved
    in the merge — a source (would cascade-delete into a dangling [^N]) or the
    target (would FK-violate into a bare INTERNAL_ERROR) — is refused upfront
    with CITES_MERGED_CHUNKS naming the offending chunk id; nothing is deleted."""
    project = seeded_project["project"]
    chunks = _doc_chunk_ids(project, seeded_project["doc_a"])
    c0, c1 = chunks[0], chunks[1]

    target = _save(project, tmp_path, capsys, name="t", title="Keep", cite=c0)
    src = _save(project, tmp_path, capsys, name="s", title="Dup", cite=c1)

    # Cite a body chunk owned by whichever involved finding we're exercising.
    involved = target if cite_role == "target" else src
    bad_chunk = _finding_chunk_ids(project, involved)[0]

    merged_file = tmp_path / "merged.md"
    merged_file.write_text(
        f"# C\n\nDoc cite[^{c0}] plus a finding chunk[^{bad_chunk}].",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        merge_findings.main([
            "--project", project,
            "--from", str(src),
            "--into", str(target),
            "--body-file", str(merged_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "CITES_MERGED_CHUNKS"
    assert out["offending_chunk_ids"] == [bad_chunk]

    # The merge aborted before any deletion: both findings still exist.
    conn = open_db(project)
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id IN (?, ?)",
            (target, src),
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_merge_memory_off_foreign_source_rejected(seeded_project, tmp_path, capsys):
    """A memory-off session cannot consume a foreign session's finding, and the
    gate fires before any deletion — both findings survive."""
    project = seeded_project["project"]
    c0 = _doc_chunk_ids(project, seeded_project["doc_a"])[0]

    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        foreign_src = _seed_finding_as(conn, author)
    finally:
        conn.close()

    # A separate memory-off session owns the target but tries to fold in the
    # foreign source.
    start_session(project, memory_enabled=False)
    target = _save(project, tmp_path, capsys, name="t", title="Mine", cite=c0)

    merged_file = tmp_path / "m.md"
    merged_file.write_text(f"Body[^{c0}].", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        merge_findings.main([
            "--project", project,
            "--from", str(foreign_src),
            "--into", str(target),
            "--body-file", str(merged_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MEMORY_OFF"
    assert out["foreign_finding_ids"] == [foreign_src]

    conn = open_db(project)
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id IN (?, ?)",
            (target, foreign_src),
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_merge_memory_off_all_own(seeded_project, tmp_path, capsys):
    """A memory-off session can merge findings it authored itself."""
    project = seeded_project["project"]
    chunks = _doc_chunk_ids(project, seeded_project["doc_a"])
    c0, c1 = chunks[0], chunks[1]

    start_session(project, memory_enabled=False)
    target = _save(project, tmp_path, capsys, name="t", title="Keep", cite=c0)
    src = _save(project, tmp_path, capsys, name="s", title="Dup", cite=c1)

    merged_file = tmp_path / "merged.md"
    merged_file.write_text(f"# C\n\nTogether[^{c0}][^{c1}].", encoding="utf-8")
    merge_findings.main([
        "--project", project,
        "--from", str(src),
        "--into", str(target),
        "--body-file", str(merged_file),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["finding_id"] == target
    assert out["merged_from"] == [src]

    conn = open_db(project)
    try:
        assert conn.cursor().execute(
            "SELECT COUNT(*) FROM findings WHERE finding_id = ?", (src,),
        ).fetchone()[0] == 0
    finally:
        conn.close()
