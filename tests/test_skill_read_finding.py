"""Smoke test for skill/scripts/read_finding.py."""

from __future__ import annotations

import json

import pytest

from bartleby.db.connection import open_db
from bartleby.skill_scripts import read_finding
from tests._skill_fixtures import (  # noqa: F401
    project_env,
    seed_finding,
    seeded_project,
)


def _active_session_id(project):
    from bartleby.session import ensure_active_session
    return ensure_active_session(project)


def _run(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        read_finding.main(argv)
    return exc.value.code, capsys.readouterr()


def test_read_finding_happy_path(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)

    conn = open_db(project)
    try:
        cited = [
            r[0] for r in conn.cursor().execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='document' "
                "AND source_id=? ORDER BY chunk_index LIMIT 2",
                (seeded_project["doc_a"],),
            )
        ]
        body = f"Claim one[^chunk:{cited[0]}] and claim two[^chunk:{cited[1]}]."
        finding_id, body_chunk_ids = seed_finding(
            conn, session_id=session_id, title="PM25 equity",
            description="who bears the brunt", body=body, cited_chunk_ids=cited,
        )
    finally:
        conn.close()

    read_finding.main([
        "--project", project, "--finding-id", f"finding:{finding_id}",
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["finding_id"] == f"finding:{finding_id}"
    assert out["title"] == "PM25 equity"
    assert out["description"] == "who bears the brunt"
    # body comes back byte-for-byte (the verbatim-echo contract).
    assert out["body"] == body
    assert out["created_at"] is not None
    assert out["session_id"] == session_id
    assert out["session_name"]
    assert out["chunk_ids"] == [f"chunk:{cid}" for cid in body_chunk_ids]

    assert [c["chunk_id"] for c in out["citations"]] == [
        f"chunk:{cid}" for cid in cited
    ]
    for c in out["citations"]:
        assert c["source_kind"] == "document"
        assert c["source_name"] == "alpha.pdf"
        assert c["file_name"] == "alpha.pdf"

    # Every [^N] resolves, so nothing dangles.
    assert out["dangling_citations"] == []


def test_read_finding_dangling_citation(seeded_project, capsys):
    """A [^N] whose cited chunk was deleted surfaces in dangling_citations."""
    from bartleby.db.chunks import delete_chunks_for

    project = seeded_project["project"]
    session_id = _active_session_id(project)

    conn = open_db(project)
    try:
        cited = [
            r[0] for r in conn.cursor().execute(
                "SELECT chunk_id FROM chunks WHERE source_kind='document' "
                "AND source_id=? ORDER BY chunk_index LIMIT 2",
                (seeded_project["doc_a"],),
            )
        ]
        body = f"Live claim[^chunk:{cited[0]}] and orphaned claim[^chunk:{cited[1]}]."
        finding_id, _ = seed_finding(
            conn, session_id=session_id, body=body, cited_chunk_ids=cited,
        )
        # Drop the cited document's chunks; ON DELETE CASCADE strips the
        # finding_citations rows pointing at them, leaving both [^chunk:N] markers
        # in the verbatim body pointing at nothing.
        delete_chunks_for(conn, "document", seeded_project["doc_a"])
    finally:
        conn.close()

    read_finding.main(["--project", project, "--finding-id", f"finding:{finding_id}"])
    out = json.loads(capsys.readouterr().out)

    # Both cited chunks belonged to the deleted document, so both [^chunk:N] markers
    # dangle and neither resolves.
    assert out["citations"] == []
    assert out["dangling_citations"] == [f"chunk:{cid}" for cid in cited]


def test_read_finding_no_citations(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)

    conn = open_db(project)
    try:
        finding_id, _ = seed_finding(conn, session_id=session_id)
    finally:
        conn.close()

    read_finding.main(["--project", project, "--finding-id", f"finding:{finding_id}"])
    out = json.loads(capsys.readouterr().out)
    assert out["citations"] == []
    assert len(out["chunk_ids"]) == 1


def test_read_finding_not_found(seeded_project, capsys):
    code, captured = _run(capsys, [
        "--project", seeded_project["project"], "--finding-id", "finding:99999",
    ])
    assert code == 1
    out = json.loads(captured.out)
    assert out["code"] == "FINDING_NOT_FOUND"


def test_read_finding_memory_off_other_session(seeded_project, capsys):
    """A memory-off session cannot read a finding authored by another session."""
    from bartleby.session import start_session

    project = seeded_project["project"]
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
            ("author", 1),
        )
        author = conn.last_insert_rowid()
        finding_id, _ = seed_finding(conn, session_id=author)
    finally:
        conn.close()

    start_session(project, memory_enabled=False)

    code, captured = _run(capsys, [
        "--project", project, "--finding-id", f"finding:{finding_id}",
    ])
    assert code == 1
    out = json.loads(captured.out)
    assert out["code"] == "MEMORY_OFF"


def test_read_finding_memory_off_own_session(seeded_project, capsys):
    """A memory-off session can still read back a finding it authored itself."""
    from bartleby.session import start_session

    project = seeded_project["project"]
    info = start_session(project, memory_enabled=False)

    conn = open_db(project)
    try:
        finding_id, _ = seed_finding(
            conn, session_id=info["session_id"], title="own", description="mine",
        )
    finally:
        conn.close()

    read_finding.main(["--project", project, "--finding-id", f"finding:{finding_id}"])
    out = json.loads(capsys.readouterr().out)
    assert out["finding_id"] == f"finding:{finding_id}"
    assert out["title"] == "own"
    assert out["session_id"] == info["session_id"]
