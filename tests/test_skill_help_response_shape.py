"""--help surfaces each script's JSON response shape (issue #47).

Every skill script documents its return contract in its module docstring's
``Output:`` block. These tests assert that ``--help`` actually prints that
docstring, so the agent can learn the response shape without running the
script against real data — and that a future script can't silently regress to
a bare ``ArgumentParser`` that hides the contract again.
"""

from __future__ import annotations

import importlib

import pytest

from bartleby.commands.skill import SCRIPTS


def _help_output(name: str, capsys) -> str:
    mod = importlib.import_module(f"bartleby.skill_scripts.{name}")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    # argparse prints help to stdout and exits 0.
    assert exc.value.code == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("name", SCRIPTS)
def test_help_prints_module_docstring(name, capsys):
    mod = importlib.import_module(f"bartleby.skill_scripts.{name}")
    assert mod.__doc__, f"{name} has no module docstring"
    out = _help_output(name, capsys)

    # The first docstring line (the one-line summary) must appear in --help.
    first_line = mod.__doc__.strip().splitlines()[0]
    assert first_line in out, f"{name} --help omits its docstring summary"

    # The return contract is documented under some 'output' heading.
    assert "output" in out.lower(), f"{name} --help omits its Output block"


@pytest.mark.parametrize(
    "name, key",
    [
        ("search", '"results"'),
        ("scan", '"matches"'),
        ("list_documents", '"documents"'),
        ("list_findings", '"findings"'),
    ],
)
def test_help_documents_divergent_top_level_keys(name, key, capsys):
    """The results/matches/documents/findings divergence the issue calls out
    is visible in each script's --help, not something to rediscover at runtime."""
    out = _help_output(name, capsys)
    assert key in out, f"{name} --help should document its top-level key {key}"
