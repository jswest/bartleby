"""Smoke test for skill/scripts/read_document.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import read_document
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def test_read_document_default_returns_both(seeded_project, capsys):
    read_document.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["document"]["file_name"] == "alpha.pdf"
    assert out["summary"] == "A summary of alpha."
    assert out["full_text"]
    assert "alpha chunk zero" in out["full_text"]


def test_read_document_summary_only(seeded_project, capsys):
    read_document.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--summary",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]
    assert out["full_text"] is None


def test_read_document_full_only(seeded_project, capsys):
    read_document.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--full",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["summary"] is None
    assert out["full_text"]


def test_read_document_too_large_without_force(seeded_project, capsys, monkeypatch):
    monkeypatch.setattr(
        "bartleby.skill_scripts.read_document.load_config",
        lambda: {"max_read_tokens": 10},
    )
    with pytest.raises(SystemExit) as exc:
        read_document.main([
            "--project", seeded_project["project"],
            "--document", str(seeded_project["doc_a"]),
            "--full",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "DOCUMENT_TOO_LARGE"
    assert out["token_count"] == 1000
    assert out["max_read_tokens"] == 10


def test_read_document_force_bypasses_gate(seeded_project, capsys, monkeypatch):
    monkeypatch.setattr(
        "bartleby.skill_scripts.read_document.load_config",
        lambda: {"max_read_tokens": 10},
    )
    read_document.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--full", "--force",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["full_text"]
