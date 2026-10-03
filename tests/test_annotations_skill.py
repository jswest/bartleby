"""Skill-side finding annotations (#732): annotate_finding, read_finding's
``annotations``, list_findings' ``annotation_count``, and the memory wall."""

from __future__ import annotations

import json

import pytest

from bartleby.db.annotations import count_annotations, insert_annotation
from bartleby.db.connection import open_db
from bartleby.session import start_session
from bartleby.skill_scripts import (
    annotate_finding,
    delete_annotation,
    delete_finding,
    edit_finding,
    list_findings,
    merge_findings,
    read_finding,
)
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seed_finding,
    seed_finding_via_main,
    seeded_project,
    unprefix,
)


def _run(script, args, capsys) -> dict:
    script.main(args)
    return json.loads(capsys.readouterr().out)


def _run_err(script, args, capsys) -> dict:
    with pytest.raises(SystemExit) as exc:
        script.main(args)
    assert exc.value.code == 1
    return json.loads(capsys.readouterr().out)


def _annotate(project, finding_id, *extra) -> list[str]:
    return ["--project", project, "--finding-id", finding_id, *extra]


def _foreign_finding(project) -> int:
    """A finding authored by another (memory-on) session."""
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        finding_id, _ = seed_finding(conn, conn.last_insert_rowid(), body="Claim one.")
    finally:
        conn.close()
    return finding_id


def test_annotate_whole_finding_and_read_back(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    saved = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )
    fid = saved["finding_id"]

    out = _run(annotate_finding, _annotate(project, fid, "--body", "Caveat."), capsys)
    assert out["annotation_id"].startswith("annotation:")
    assert out["finding_id"] == fid
    assert out["body"] == "Caveat."
    assert out["anchor"] is None
    assert out["chunk_id"] is None
    assert out["is_human_author"] is False

    read = _run(read_finding, ["--project", project, "--finding-id", fid], capsys)
    assert read["body"] == saved["body"]  # the finding body is never touched
    [note] = read["annotations"]
    assert note["annotation_id"] == out["annotation_id"]
    assert note["anchor"] is None
    assert note["anchor_found"] is True
    assert note["is_human_author"] is False
    # Stamped with the authoring session (here, the same one that saved it).
    assert note["session_id"] == read["session_id"]


def test_annotate_anchored_with_chunk_and_body_file(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    saved = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )
    fid = saved["finding_id"]
    a, _ = saved["_chunks"]
    note_file = tmp_path / "note.md"
    note_file.write_text("Wrong: costs $5 (see `x`).", encoding="utf-8")

    out = _run(annotate_finding, _annotate(
        project, fid, "--body-file", str(note_file),
        "--quote", "Claim", "--quote-suffix", " two", "--chunk-id", f"chunk:{a}",
    ), capsys)
    assert out["body"] == "Wrong: costs $5 (see `x`)."
    assert out["anchor"] == {"exact": "Claim", "prefix": None, "suffix": " two"}
    assert out["chunk_id"] == f"chunk:{a}"

    read = _run(read_finding, ["--project", project, "--finding-id", fid], capsys)
    [note] = read["annotations"]
    assert note["anchor"] == out["anchor"]
    assert note["anchor_found"] is True
    assert note["chunk_id"] == f"chunk:{a}"


def test_annotate_anchor_not_found(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run_err(annotate_finding, _annotate(
        project, fid, "--body", "x", "--quote", "not in the body",
    ), capsys)
    assert out["code"] == "ANCHOR_NOT_FOUND"


@pytest.mark.parametrize("flag", ["--quote-prefix", "--quote-suffix"])
def test_annotate_affix_requires_quote(seeded_project, tmp_path, capsys, flag):
    project = seeded_project["project"]
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run_err(annotate_finding, _annotate(
        project, fid, "--body", "x", flag, "Claim",
    ), capsys)
    assert out["code"] == "QUOTE_REQUIRED"


def test_annotate_unknown_finding_and_chunk(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    out = _run_err(annotate_finding, _annotate(
        project, "finding:9999", "--body", "x",
    ), capsys)
    assert out["code"] == "FINDING_NOT_FOUND"

    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run_err(annotate_finding, _annotate(
        project, fid, "--body", "x", "--chunk-id", "chunk:99999",
    ), capsys)
    assert out["code"] == "UNKNOWN_CHUNK"


def test_annotate_memory_off_foreign_refused_own_allowed(
    seeded_project, tmp_path, capsys
):
    project = seeded_project["project"]
    foreign = _foreign_finding(project)
    start_session(project, memory_enabled=False)

    out = _run_err(annotate_finding, _annotate(
        project, f"finding:{foreign}", "--body", "x",
    ), capsys)
    assert out["code"] == "MEMORY_OFF"
    conn = open_db(project)
    try:
        assert count_annotations(conn, [foreign]) == {foreign: 0}
    finally:
        conn.close()

    own = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run(annotate_finding, _annotate(project, own, "--body", "mine"), capsys)
    assert out["finding_id"] == own


def test_read_finding_anchor_found_false_after_edit(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    saved = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )
    fid = saved["finding_id"]
    a, _ = saved["_chunks"]
    _run(annotate_finding, _annotate(
        project, fid, "--body", "Wrong.", "--quote", "Claim two",
    ), capsys)

    new_body = tmp_path / "edited.md"
    new_body.write_text(f"# Seed\n\nOnly claim one[^chunk:{a}].", encoding="utf-8")
    _run(edit_finding, [
        "--project", project, "--finding-id", fid, "--body-file", str(new_body),
    ], capsys)

    read = _run(read_finding, ["--project", project, "--finding-id", fid], capsys)
    [note] = read["annotations"]
    assert note["anchor"]["exact"] == "Claim two"
    assert note["anchor_found"] is False


def test_list_findings_annotation_count(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    noted = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="Noted", description="D",
    )["finding_id"]
    bare = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="Bare", description="D",
    )["finding_id"]
    for text in ("one", "two"):
        _run(annotate_finding, _annotate(project, noted, "--body", text), capsys)

    for extra in ([], ["--brief"]):
        out = _run(list_findings, ["--project", project, *extra], capsys)
        counts = {f["finding_id"]: f["annotation_count"] for f in out["findings"]}
        assert counts == {noted: 2, bare: 0}


def _annotation_count(project, finding_id: int) -> int:
    conn = open_db(project)
    try:
        return count_annotations(conn, [finding_id])[finding_id]
    finally:
        conn.close()


def test_delete_annotation_happy_path(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    note = _run(annotate_finding, _annotate(project, fid, "--body", "Gone."), capsys)

    out = _run(delete_annotation, [
        "--project", project, "--annotation-id", note["annotation_id"],
    ], capsys)
    assert out["status"] == "deleted"
    assert out["annotation_id"] == note["annotation_id"]
    assert out["finding_id"] == fid
    assert out["body"] == "Gone."
    assert _annotation_count(project, unprefix(fid)) == 0


def test_delete_annotation_not_found(seeded_project, capsys):
    out = _run_err(delete_annotation, [
        "--project", seeded_project["project"], "--annotation-id", "annotation:999",
    ], capsys)
    assert out["code"] == "ANNOTATION_NOT_FOUND"


def test_delete_annotation_memory_off_foreign_refused_own_allowed(
    seeded_project, tmp_path, capsys
):
    project = seeded_project["project"]
    foreign = _foreign_finding(project)
    # Annotate the foreign finding from a memory-on session first.
    foreign_note = _run(annotate_finding, _annotate(
        project, f"finding:{foreign}", "--body", "theirs",
    ), capsys)["annotation_id"]

    start_session(project, memory_enabled=False)
    out = _run_err(delete_annotation, [
        "--project", project, "--annotation-id", foreign_note,
    ], capsys)
    assert out["code"] == "MEMORY_OFF"
    assert _annotation_count(project, foreign) == 1  # row survives

    own = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    own_note = _run(annotate_finding, _annotate(project, own, "--body", "mine"), capsys)
    out = _run(delete_annotation, [
        "--project", project, "--annotation-id", own_note["annotation_id"],
    ], capsys)
    assert out["finding_id"] == own


def test_annotate_rejects_quote_splitting_a_citation_marker(
    seeded_project, tmp_path, capsys
):
    project = seeded_project["project"]
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run_err(annotate_finding, _annotate(
        project, fid, "--body", "x", "--quote", "one[^chunk:",
    ), capsys)
    assert out["code"] == "ANCHOR_NOT_FOUND"
    assert "citation marker" in out["error"]
    assert _annotation_count(project, unprefix(fid)) == 0


def test_annotate_strips_body(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    out = _run(annotate_finding, _annotate(project, fid, "--body", "  note \n"), capsys)
    assert out["body"] == "note"
    assert out["anchor_found"] is True and out["created_at"]


def test_memory_off_read_hides_foreign_agent_notes(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    start_session(project, memory_enabled=False)
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    _run(annotate_finding, _annotate(project, fid, "--body", "mine"), capsys)
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES ('other', 1)"
        )
        other = conn.last_insert_rowid()
        insert_annotation(conn, finding_id=unprefix(fid), body="theirs",
                          is_human_author=False, session_id=other)
        insert_annotation(conn, finding_id=unprefix(fid), body="human",
                          is_human_author=True)
    finally:
        conn.close()

    read = _run(read_finding, ["--project", project, "--finding-id", fid], capsys)
    assert [a["body"] for a in read["annotations"]] == ["mine", "human"]
    listed = _run(list_findings, ["--project", project], capsys)
    [row] = listed["findings"]
    assert row["annotation_count"] == 2


def test_merge_moves_source_annotations_to_target(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    into = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="Into", description="D",
    )
    src = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="Src", description="D",
    )["finding_id"]
    conn = open_db(project)
    try:
        insert_annotation(conn, finding_id=unprefix(src), body="$3M is wrong",
                          is_human_author=True, anchor_exact="Claim one")
    finally:
        conn.close()
    body_file = tmp_path / "merged.md"
    body_file.write_text(into["body"].replace("Claim one", "Merged claim"))

    out = _run(merge_findings, [
        "--project", project, "--from", src, "--into", into["finding_id"],
        "--body-file", str(body_file),
    ], capsys)
    assert out["annotations_moved"] == 1
    read = _run(read_finding, [
        "--project", project, "--finding-id", into["finding_id"],
    ], capsys)
    [note] = read["annotations"]
    assert note["body"] == "$3M is wrong"
    assert note["anchor_found"] is False  # anchored to the old source body


def test_delete_finding_reports_annotations_dropped(seeded_project, tmp_path, capsys):
    project = seeded_project["project"]
    fid = seed_finding_via_main(
        seeded_project, tmp_path, capsys, title="T", description="D",
    )["finding_id"]
    for text in ("one", "two"):
        _run(annotate_finding, _annotate(project, fid, "--body", text), capsys)
    out = _run(delete_finding, ["--project", project, "--finding-id", fid], capsys)
    assert out["annotations_dropped"] == 2
