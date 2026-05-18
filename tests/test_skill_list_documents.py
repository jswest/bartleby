"""Smoke test for skill/scripts/list_documents.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import list_documents
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def _run(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        list_documents.main(argv)
    return exc.value.code, capsys.readouterr()


def test_list_documents_happy_path(seeded_project, capsys):
    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 2
    by_name = {d["file_name"]: d for d in out["documents"]}
    assert by_name["alpha.pdf"]["has_summary"] is True
    assert by_name["alpha.pdf"]["chunk_count"] == 4
    assert by_name["beta.txt"]["has_summary"] is False
    assert by_name["beta.txt"]["chunk_count"] == 2


def test_list_documents_limit_and_offset(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"],
        "--limit", "1", "--offset", "1",
    ])
    out = json.loads(capsys.readouterr().out)
    assert len(out["documents"]) == 1
    assert out["total"] == 2
