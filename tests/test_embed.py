"""Tests for `bartleby embed` (CLI + module)."""

from __future__ import annotations

import json
import subprocess

import pytest

from bartleby.commands import embed as embed_cmd
from bartleby.db.schema import EMBEDDING_DIM


def test_embed_command_prints_array_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(
        "bartleby.commands.embed.embed_texts",
        lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts],
    )
    embed_cmd.main("hello")
    out = capsys.readouterr().out
    vec = json.loads(out)
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM


def test_embed_command_rejects_empty_text(capsys):
    with pytest.raises(SystemExit) as exc:
        embed_cmd.main("   ")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "non-empty" in err


def test_embed_command_subprocess_listform(monkeypatch):
    """End-to-end via the installed `bartleby` CLI.

    Verifies the skill's calling pattern works: argv-passed text reaches the
    embedder without shell interpretation, and the response parses as a
    correctly-sized JSON array.
    """
    proc = subprocess.run(
        ["uv", "run", "bartleby", "embed", "hello world; rm -rf /"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    vec = json.loads(proc.stdout)
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec)
