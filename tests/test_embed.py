"""Tests for `bartleby embed` (CLI + module)."""

from __future__ import annotations

import json

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


def test_embed_command_passes_argv_text_through_verbatim(monkeypatch, capsys):
    """The skill's calling pattern: list-form ``["bartleby", "embed", query]``
    hands the query to the embedder as ``argv``, so shell metacharacters in it
    are inert data, never interpreted (SPEC §5.5).

    We assert the property without spawning the real CLI — and so without
    loading the BAAI model — by monkeypatching ``embed_texts`` to record exactly
    what ``main`` forwards. A regression that mangled, split, or shell-expanded
    the argument (e.g. stripping past the ``;``) would change the recorded text
    and fail here. The cross-process, no-shell-interpretation guarantee of
    list-form ``subprocess.run`` is Python's, not ours to re-test per run.
    """
    adversarial = "hello world; rm -rf /"
    seen: list[list[str]] = []

    def _fake(texts):
        seen.append(texts)
        return [[0.2] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr("bartleby.commands.embed.embed_texts", _fake)

    embed_cmd.main(adversarial)

    # The full argument reached the embedder intact — not split on the ';'.
    assert seen == [[adversarial]]
    vec = json.loads(capsys.readouterr().out)
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec)
