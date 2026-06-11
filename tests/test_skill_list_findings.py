"""Smoke test for skill/scripts/list_findings.py."""

from __future__ import annotations

import json

import pytest

from bartleby.db.chunks import ChunkInput, insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.skill_scripts import list_findings
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def _emb() -> list[float]:
    return [0.01 * i for i in range(EMBEDDING_DIM)]


def _seed_finding(conn, *, session_id, title, description, body="body",
                  cited_chunk_ids=()):
    """Insert a finding (+ optional citations) directly, returning its id."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO findings (session_id, title, description, body) "
        "VALUES (?, ?, ?, ?)",
        (session_id, title, description, body),
    )
    finding_id = conn.last_insert_rowid()
    insert_finding_chunks(conn, finding_id, [
        ChunkInput(text=body, embedding=_emb(), chunk_index=0),
    ])
    for chunk_id in cited_chunk_ids:
        cur.execute(
            "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
            (finding_id, chunk_id),
        )
    return finding_id


def _active_session_id(project):
    """The session the runner auto-resolves (created by seeded_project access)."""
    from bartleby.session import ensure_active_session
    return ensure_active_session(project)


def test_list_findings_empty(seeded_project, capsys):
    list_findings.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 0
    assert out["findings"] == []
    assert out["hint"] is None


def test_list_findings_happy_path(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)

    conn = open_db(project)
    try:
        doc_chunk = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id=? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
        _seed_finding(conn, session_id=session_id, title="First",
                      description="oldest", cited_chunk_ids=[doc_chunk])
        _seed_finding(conn, session_id=session_id, title="Second",
                      description="newest", cited_chunk_ids=[])
    finally:
        conn.close()

    list_findings.main(["--project", project])
    out = json.loads(capsys.readouterr().out)

    assert out["total"] == 2
    # Newest-first ordering (finding_id DESC).
    assert [f["title"] for f in out["findings"]] == ["Second", "First"]
    first, second = out["findings"][1], out["findings"][0]
    assert first["citation_count"] == 1
    assert second["citation_count"] == 0
    assert first["description"] == "oldest"
    assert first["session_name"]
    assert first["created_at"] is not None


def test_list_findings_brief_projects_three_fields(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)

    conn = open_db(project)
    try:
        doc_chunk = conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id=? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
        _seed_finding(conn, session_id=session_id, title="Cited",
                      description="hook", cited_chunk_ids=[doc_chunk])
    finally:
        conn.close()

    list_findings.main(["--project", project, "--brief"])
    out = json.loads(capsys.readouterr().out)

    assert out["findings"]
    for f in out["findings"]:
        assert set(f) == {"finding_id", "title", "citation_count"}
    assert out["findings"][0]["citation_count"] == 1
    assert out["total"] == 1  # envelope unchanged


def test_list_findings_pagination_hint(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)

    conn = open_db(project)
    try:
        for i in range(3):
            _seed_finding(conn, session_id=session_id, title=f"f{i}",
                          description="x")
    finally:
        conn.close()

    list_findings.main(["--project", project, "--limit", "2", "--offset", "0"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 3
    assert len(out["findings"]) == 2
    assert out["hint"] == "Showing 1-2 of 3. Pass --offset 2 to continue."

    # Last page leaves no hint.
    list_findings.main(["--project", project, "--limit", "2", "--offset", "2"])
    out = json.loads(capsys.readouterr().out)
    assert len(out["findings"]) == 1
    assert out["hint"] is None


@pytest.mark.parametrize("bad", [
    ["--limit", "0"],     # positive_int rejects < 1
    ["--limit", "-5"],
    ["--offset", "-1"],   # nonneg_int rejects < 0
])
def test_list_findings_out_of_range_pagination_rejected(seeded_project, capsys, bad):
    # Shared positive/non-negative validators (issue #403) reject at parse time:
    # the JSON usage envelope is emitted before any query runs.
    with pytest.raises(SystemExit) as exc:
        list_findings.main(["--project", seeded_project["project"], *bad])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


def test_list_findings_memory_off_scopes_to_own_session(seeded_project, capsys):
    """Memory-off lists only the active session's findings, hiding others'."""
    from bartleby.session import start_session

    project = seeded_project["project"]
    # A finding authored by some *other* session.
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        _seed_finding(conn, session_id=author, title="hidden", description="x")
    finally:
        conn.close()

    # Start a memory-off session and give it one finding of its own.
    info = start_session(project, memory_enabled=False)
    conn = open_db(project)
    try:
        _seed_finding(conn, session_id=info["session_id"],
                      title="mine", description="y")
    finally:
        conn.close()

    list_findings.main(["--project", project])
    out = json.loads(capsys.readouterr().out)
    # Only this session's finding is visible; the other session's is excluded,
    # and total reflects the scoped set.
    assert out["total"] == 1
    assert [f["title"] for f in out["findings"]] == ["mine"]
