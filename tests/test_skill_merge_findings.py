"""Smoke test for skill/scripts/merge_findings.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import merge_findings, save_finding
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
