"""Smoke test for skill_scripts/save_date.py.

save_date writes ``summaries.authored_date`` — the column every read path
(list_documents, scan/search, the date filters) actually reads. The
read-path test below asserts the written value surfaces through
list_documents.
"""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import save_date, list_documents
from bartleby.db.connection import open_db
from tests._skill_fixtures import (  # noqa: F401
    project_env,
    seeded_project,
)


def _authored_date(project, document_id):
    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT authored_date FROM summaries WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_save_date_writes_and_returns_old_and_new(seeded_project, capsys):
    # doc_a starts with a summary but NULL authored_date.
    save_date.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--date", "2024-03-08",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["document_id"] == seeded_project["doc_a"]
    assert out["old_authored_date"] is None
    assert out["new_authored_date"] == "2024-03-08"
    assert _authored_date(seeded_project["project"], seeded_project["doc_a"]) == "2024-03-08"


def test_save_date_visible_through_list_documents(seeded_project, capsys):
    """The written value surfaces through a real agent-facing read path."""
    save_date.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--date", "2019-11-21",
    ])
    capsys.readouterr()  # drop save_date's output

    list_documents.main(["--project", seeded_project["project"]])
    listed = json.loads(capsys.readouterr().out)
    by_id = {d["id"]: d for d in listed["documents"]}
    assert by_id[seeded_project["doc_a"]]["authored_date"] == "2019-11-21"


def test_save_date_rejects_malformed(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        save_date.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
            "--date", "Q3 2024",
        ])
    assert exc.value.code != 0
    err = json.loads(capsys.readouterr().out)
    assert err["code"] == "INVALID_DATE"
    # The bad write left the stored value untouched (still NULL).
    assert _authored_date(seeded_project["project"], seeded_project["doc_a"]) is None


def test_save_date_rejects_impossible_calendar_date(seeded_project, capsys):
    with pytest.raises(SystemExit):
        save_date.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
            "--date", "2024-13-01",
        ])
    err = json.loads(capsys.readouterr().out)
    assert err["code"] == "INVALID_DATE"


def test_save_date_clear_nulls_the_field(seeded_project, capsys):
    # First set it, then clear it.
    save_date.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--date", "2024-03-08",
    ])
    capsys.readouterr()

    save_date.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--clear",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["old_authored_date"] == "2024-03-08"
    assert out["new_authored_date"] is None
    assert _authored_date(seeded_project["project"], seeded_project["doc_a"]) is None


def test_save_date_no_summary_is_rejected(seeded_project, capsys):
    # doc_b has no summary row to carry a date.
    with pytest.raises(SystemExit):
        save_date.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_b"]),
            "--date", "2024-03-08",
        ])
    err = json.loads(capsys.readouterr().out)
    assert err["code"] == "NO_SUMMARY"


def test_save_date_unknown_document_is_rejected(seeded_project, capsys):
    with pytest.raises(SystemExit):
        save_date.main([
            "--project", seeded_project["project"],
            "--document", "999999",
            "--date", "2024-03-08",
        ])
    err = json.loads(capsys.readouterr().out)
    assert err["code"] == "DOCUMENT_NOT_FOUND"


def test_save_date_requires_date_or_clear(seeded_project, capsys):
    with pytest.raises(SystemExit):
        save_date.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
        ])
