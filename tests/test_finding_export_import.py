"""Tests for `bartleby finding export` / `import` (out-of-band MD share, #518).

Covers the round trip: export a finding to a self-describing artifact, import it
into a *fresh* corpus, and assert (a) provenance lands in the body, (b) corpus
citations degrade to inert `[corpus: file · page]` markers rather than live
chunk links, and (c) the imported finding is a normal local finding.
"""

from __future__ import annotations

import json

import pytest

import bartleby.project
from bartleby.commands import finding as finding_cmd
from bartleby.db.connection import open_db
from bartleby.skill_scripts import save_finding
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
)


def _save_a_finding(seeded_project, tmp_path, capsys, *, with_page=False):
    """Save a finding citing real corpus chunks; return its parsed response."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        if with_page:
            from bartleby.db.chunks import ChunkInput, insert_document_chunks
            from bartleby.db.schema import EMBEDDING_DIM

            emb = [0.01 * i for i in range(EMBEDDING_DIM)]
            insert_document_chunks(conn, seeded_project["doc_a"], [
                ChunkInput(text="paged chunk", embedding=emb, chunk_index=9,
                           section_heading=None, page_number=7, content_type="text"),
            ])
            cited = [cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='document' "
                "AND source_id=? AND chunk_index=9",
                (seeded_project["doc_a"],),
            ).fetchone()[0]]
        else:
            cited = [r[0] for r in cur.execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='document' "
                "AND source_id = ? ORDER BY chunk_index LIMIT 2",
                (seeded_project["doc_a"],),
            ).fetchall()]
    finally:
        conn.close()

    body = "# Finding\n\n" + " ".join(f"Claim[^{c}]." for c in cited)
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "Air quality gaps",
        "--description", "Where PM2.5 monitoring is thin.",
        "--body-file", str(body_file),
    ])
    return json.loads(capsys.readouterr().out)


def test_export_writes_front_matter_and_inert_citations(
    seeded_project, tmp_path, capsys
):
    saved = _save_a_finding(seeded_project, tmp_path, capsys, with_page=True)
    out_path = tmp_path / "artifact.md"
    finding_cmd.export(
        finding_id=saved["finding_id"],
        project=seeded_project["project"],
        out=str(out_path),
    )
    text = out_path.read_text(encoding="utf-8")

    # YAML front-matter carries title/description + provenance.
    parsed = finding_cmd.parse_artifact(text)
    assert parsed["title"] == "Air quality gaps"
    assert parsed["description"] == "Where PM2.5 monitoring is thin."
    assert parsed["provenance"]["source_corpus"] == seeded_project["project"]
    assert parsed["provenance"]["source_finding_id"] == saved["finding_id"]
    assert parsed["provenance"]["exported_on"]

    # The live [^N] chunk marker is gone, replaced by an inert file · page marker.
    assert "[^" not in parsed["body"]
    assert "[corpus: alpha.pdf · p.7]" in parsed["body"]


def test_round_trip_import_into_fresh_corpus(seeded_project, tmp_path, capsys):
    saved = _save_a_finding(seeded_project, tmp_path, capsys)
    out_path = tmp_path / "artifact.md"
    finding_cmd.export(
        finding_id=saved["finding_id"],
        project=seeded_project["project"],
        out=str(out_path),
    )
    capsys.readouterr()

    # A *fresh* corpus with none of the source chunk_ids.
    bartleby.project.create_project("fresh")
    finding_cmd.import_(path=str(out_path), project="fresh")

    conn = open_db("fresh")
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT finding_id, title, description, body FROM findings"
        ).fetchall()
        assert len(rows) == 1
        finding_id, title, description, body = rows[0]
        assert title == "Air quality gaps"
        assert description == "Where PM2.5 monitoring is thin."

        # Provenance baked into the body text (no schema column for it).
        assert "Imported from" in body
        assert f"orig. finding #{saved['finding_id']}" in body
        assert f"corpus `{seeded_project['project']}`" in body

        # Corpus citations are inert markers carrying file metadata, NOT live
        # [^N] chunk links, and no live citation rows were written.
        assert "[corpus: alpha.pdf]" in body
        assert "[^" not in body
        n_cites = cur.execute(
            "SELECT COUNT(*) FROM finding_citations WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()[0]
        assert n_cites == 0

        # The body was chunked + embedded like any local finding.
        n_chunks = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_kind='finding' "
            "AND source_id = ?",
            (finding_id,),
        ).fetchone()[0]
        assert n_chunks >= 1
    finally:
        conn.close()


def test_imported_finding_reads_back_like_a_local_one(
    seeded_project, tmp_path, capsys
):
    """An imported finding is reachable through the normal read path."""
    from bartleby.skill_scripts import read_finding

    saved = _save_a_finding(seeded_project, tmp_path, capsys)
    out_path = tmp_path / "artifact.md"
    finding_cmd.export(
        finding_id=saved["finding_id"], project=seeded_project["project"],
        out=str(out_path),
    )
    capsys.readouterr()

    bartleby.project.create_project("fresh")
    finding_cmd.import_(path=str(out_path), project="fresh")
    capsys.readouterr()

    conn = open_db("fresh")
    try:
        new_id = conn.cursor().execute("SELECT finding_id FROM findings").fetchone()[0]
    finally:
        conn.close()

    read_finding.main(["--project", "fresh", "--finding-id", str(new_id)])
    read_out = json.loads(capsys.readouterr().out)
    assert read_out["title"] == "Air quality gaps"
    assert "Imported from" in read_out["body"]
    # No live citations resolved (the corpus markers are inert).
    assert read_out["citations"] == []


def test_export_missing_finding_errors(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        finding_cmd.export(finding_id=999999, project=seeded_project["project"],
                           out=None)
    assert exc.value.code == 1


def test_import_rejects_non_artifact(seeded_project, tmp_path, capsys):
    bad = tmp_path / "plain.md"
    bad.write_text("# just markdown, no front matter\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        finding_cmd.import_(path=str(bad), project=seeded_project["project"])
    assert exc.value.code == 1


def test_export_default_filename_slug(seeded_project, tmp_path, capsys, monkeypatch):
    """Omitting --out writes <slug>.md in the cwd, mirroring the web download."""
    saved = _save_a_finding(seeded_project, tmp_path, capsys)
    monkeypatch.chdir(tmp_path)
    finding_cmd.export(finding_id=saved["finding_id"],
                       project=seeded_project["project"], out=None)
    assert (tmp_path / "air-quality-gaps.md").is_file()


def test_finding_to_finding_citation_inlines_title(
    seeded_project, tmp_path, capsys
):
    """A [^N] citing a finding-kind chunk degrades to a titled inert marker."""
    # First finding (cites a document chunk).
    first = _save_a_finding(seeded_project, tmp_path, capsys)
    conn = open_db(seeded_project["project"])
    try:
        finding_chunk = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='finding' "
            "AND source_id = ? LIMIT 1",
            (first["finding_id"],),
        ).fetchone()[0]
        doc_chunk = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()

    # Second finding cites both a document chunk (required) and the first
    # finding's chunk.
    body = f"Builds on prior work[^{finding_chunk}], grounded in evidence[^{doc_chunk}]."
    body_file = tmp_path / "second.md"
    body_file.write_text(body, encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "Second finding", "--description", "d",
        "--body-file", str(body_file),
    ])
    second = json.loads(capsys.readouterr().out)

    out_path = tmp_path / "second.artifact.md"
    finding_cmd.export(finding_id=second["finding_id"],
                       project=seeded_project["project"], out=str(out_path))
    artifact = out_path.read_text(encoding="utf-8")
    assert "[corpus: finding · Air quality gaps]" in artifact
    assert "[^" not in finding_cmd.parse_artifact(artifact)["body"]
