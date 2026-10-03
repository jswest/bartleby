"""Tests for `bartleby finding annotate / annotations / delete-annotation` and
annotation rendering in `finding read` / `finding export` (#733)."""

from __future__ import annotations

import sys

import pytest

from bartleby import cli
from bartleby.commands import finding as finding_cmd
from bartleby.db.annotations import insert_annotation, list_annotations
from bartleby.db.connection import open_db
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
    seed_finding,
)

BODY = "Alpha is true. Beta is false.[^chunk:{c}] Gamma holds."


def _seed(seeded_project):
    from bartleby.session import ensure_active_session

    project = seeded_project["project"]
    sid = ensure_active_session(project)
    conn = open_db(project)
    try:
        cid = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' LIMIT 1"
        ).fetchone()[0]
        fid, _ = seed_finding(
            conn, sid, title="T", description="d", body=BODY.format(c=cid),
            cited_chunk_ids=[cid],
        )
    finally:
        conn.close()
    return project, fid, cid


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["bartleby", "finding", *argv])
    cli.main()


def _rows(project, fid):
    conn = open_db(project)
    try:
        return list_annotations(conn, fid)
    finally:
        conn.close()


def test_annotate_list_delete_end_to_end(seeded_project, monkeypatch, capsys):
    project, fid, cid = _seed(seeded_project)
    _run(monkeypatch, "annotate", f"finding:{fid}", "--project", project,
         "--note", "Beta is actually true", "--quote", "Beta is false",
         "--chunk-id", f"chunk:{cid}")
    new_id = capsys.readouterr().out.strip()
    assert new_id.startswith("annotation:")
    _run(monkeypatch, "annotate", f"finding:{fid}", "--project", project,
         "--note", "overall caveat")
    capsys.readouterr()

    rows = _rows(project, fid)
    assert [r["is_human_author"] for r in rows] == [1, 1]
    assert rows[0]["session_id"] is None and rows[0]["chunk_id"] == cid

    _run(monkeypatch, "annotations", f"finding:{fid}", "--project", project)
    out = capsys.readouterr().out
    assert out.index("Beta is false") < out.index("whole finding")
    assert "human" in out and "overall caveat" in out and f"chunk:{cid}" in out

    _run(monkeypatch, "delete-annotation", new_id, "--project", project)
    assert "Deleted" in capsys.readouterr().out
    assert len(_rows(project, fid)) == 1
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "delete-annotation", new_id, "--project", project)
    assert e.value.code != 0


def test_note_file_and_anchor_not_found(seeded_project, monkeypatch, capsys, tmp_path):
    project, fid, _ = _seed(seeded_project)
    nf = tmp_path / "n.txt"
    nf.write_text("from file\n")
    _run(monkeypatch, "annotate", f"finding:{fid}", "--project", project,
         "--note-file", str(nf))
    assert _rows(project, fid)[0]["body"] == "from file"
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "annotate", f"finding:{fid}", "--project", project,
             "--note", "x", "--quote", "not in body")
    assert e.value.code != 0
    assert len(_rows(project, fid)) == 1


def _capture_errors(monkeypatch) -> list[str]:
    errors: list[str] = []
    monkeypatch.setattr(finding_cmd.console, "error", errors.append)
    return errors


def test_prefix_without_quote_is_rejected(seeded_project, monkeypatch):
    project, fid, _ = _seed(seeded_project)
    errors = _capture_errors(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "annotate", f"finding:{fid}", "--project", project,
             "--note", "x", "--quote-prefix", "Alpha")
    assert e.value.code != 0
    assert "require anchor_exact" in errors[0]
    assert _rows(project, fid) == []


def test_bare_id_rejected(seeded_project, monkeypatch):
    project, fid, _ = _seed(seeded_project)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "annotations", str(fid), "--project", project)


def _annotate_direct(project, fid, **kw):
    conn = open_db(project)
    try:
        return insert_annotation(conn, finding_id=fid, **kw)
    finally:
        conn.close()


def test_read_renders_anchored_whole_and_stale(seeded_project, capsys):
    project, fid, _ = _seed(seeded_project)
    _annotate_direct(project, fid, body="check Beta", is_human_author=True,
                     anchor_exact="Beta is false")
    _annotate_direct(project, fid, body="general", is_human_author=False)
    _annotate_direct(project, fid, body="will go stale", is_human_author=True,
                     anchor_exact="Gamma holds")
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "UPDATE findings SET body = replace(body, 'Gamma holds', 'Delta') "
            "WHERE finding_id = ?", (fid,))
    finally:
        conn.close()

    finding_cmd.read(finding_id=fid, project=project, json_out=False, render=False)
    out = capsys.readouterr().out
    # Inline marker lands right after the quoted span (offsets survive the citation rewrite).
    assert "Beta is false[✎1].[^1]" in out
    assert "## Annotations" in out
    assert "✎1 · human" in out
    assert "✎2 · agent" in out and "whole finding" in out
    assert "annotation anchor no longer matches the text" in out
    assert "✎3" in out and "[✎3]" not in out  # stale: no inline marker


def test_read_without_annotations_has_no_section(seeded_project, capsys):
    project, fid, _ = _seed(seeded_project)
    finding_cmd.read(finding_id=fid, project=project, json_out=False, render=False)
    assert "Annotations" not in capsys.readouterr().out


def test_export_carries_annotations(seeded_project, tmp_path):
    project, fid, _ = _seed(seeded_project)
    _annotate_direct(project, fid, body="human correction", is_human_author=True,
                     anchor_exact="Alpha is true")
    out = tmp_path / "f.md"
    finding_cmd.export(finding_id=fid, project=project, out=str(out))
    text = out.read_text()
    assert "## Annotations" in text and "human correction" in text


def test_annotate_rejects_quote_splitting_a_citation_marker(
    seeded_project, monkeypatch
):
    project, fid, _ = _seed(seeded_project)
    errors = _capture_errors(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "annotate", f"finding:{fid}", "--project", project,
             "--note", "x", "--quote", "false.[^chunk:")
    assert e.value.code != 0
    assert "citation marker" in errors[0]
    assert _rows(project, fid) == []


def test_export_import_round_trip_drops_annotations(seeded_project, tmp_path):
    import bartleby.project

    project, fid, _ = _seed(seeded_project)
    _annotate_direct(project, fid, body="human correction", is_human_author=True,
                     anchor_exact="Alpha is true")
    out = tmp_path / "f.md"
    finding_cmd.export(finding_id=fid, project=project, out=str(out))
    text = out.read_text()
    assert "✎" not in text  # export emits no inline markers, so no numbering

    bartleby.project.create_project("fresh")
    finding_cmd.import_(path=str(out), project="fresh")
    conn = open_db("fresh")
    try:
        [(body,)] = conn.cursor().execute("SELECT body FROM findings").fetchall()
    finally:
        conn.close()
    assert "Annotations" not in body and "human correction" not in body
    assert body.rstrip().endswith("Gamma holds.")
