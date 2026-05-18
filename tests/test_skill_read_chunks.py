"""Smoke test for skill/scripts/read_chunks.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import read_chunks
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def test_read_chunks_happy_path(seeded_project, capsys):
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["document"]["file_name"] == "alpha.pdf"
    assert out["total"] == 4
    assert len(out["chunks"]) == 4
    indexes = [c["chunk_index"] for c in out["chunks"]]
    assert indexes == [0, 1, 2, 3]
    assert out["chunks"][0]["section_heading"] == "Intro"


def test_read_chunks_pagination(seeded_project, capsys):
    read_chunks.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--offset", "1", "--limit", "2",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 4
    assert [c["chunk_index"] for c in out["chunks"]] == [1, 2]


def test_read_chunks_unknown_document(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document", "999",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "DOCUMENT_NOT_FOUND"
