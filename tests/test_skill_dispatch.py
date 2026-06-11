"""Contract tests for the ``bartleby skill`` dispatcher (issue #401).

A bare ``bartleby skill`` (no script name) must behave like every other skill
error path: a ``{"error", "code"}`` JSON envelope on stdout, prose to stderr
only, and a non-zero exit. ``-h/--help`` must still exit 0 with help text.
"""

from __future__ import annotations

import json

import pytest

from bartleby.commands.skill import SCRIPTS, dispatch


def test_bare_invocation_emits_error_envelope(capsys):
    """No script name -> JSON envelope on stdout + exit 1; prose to stderr."""
    with pytest.raises(SystemExit) as exc:
        dispatch([])
    assert exc.value.code == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {"error", "code"}
    assert payload["code"] == "MISSING_SKILL"
    # Usage prose goes to stderr only, never stdout.
    assert "usage: bartleby skill" not in captured.out
    assert "usage: bartleby skill" in captured.err


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flag_exits_zero_with_help_text(flag, capsys):
    """``-h/--help`` stays exit 0 with usage help on stdout (no envelope)."""
    dispatch([flag])  # returns normally, no SystemExit
    captured = capsys.readouterr()
    assert "usage: bartleby skill" in captured.out


def test_unknown_skill_still_emits_error_envelope(capsys):
    """The pre-existing unknown-name path keeps its envelope + exit 1."""
    with pytest.raises(SystemExit) as exc:
        dispatch(["not_a_real_script"])
    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "UNKNOWN_SKILL"


def test_scripts_tuple_nonempty():
    assert SCRIPTS, "the dispatcher must advertise at least one script"
