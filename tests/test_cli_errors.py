"""Tests for the CLI's expected-error → clean-exit contract (#491).

Every expected user error must exit 1 with a one-line message on *stderr*
(never a traceback), and wizard prompts validate their own bounds. The shared
Rich Console binds its stream at import time, so — following the repo idiom
(see test_scribe's reasoning-effort tests) — we capture by monkeypatching the
``console.error`` / ``console.warn`` symbol at the use site rather than via
capsys.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bartleby.project
from bartleby import cli
from bartleby.commands import config as config_cmd
from bartleby.commands import logs as logs_cmd
from bartleby.commands import project as project_cmd
from bartleby.ingest import resolve


@pytest.fixture
def project():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    bartleby.project.create_project("alpha")
    return "alpha"


def _capture_error(monkeypatch, module):
    """Redirect ``module.console.error`` into a list and return it."""
    errors: list[str] = []
    monkeypatch.setattr(module.console, "error", lambda m: errors.append(m))
    return errors


# ---- scribe: expected errors exit 1 on stderr, never a traceback -------------


def _scribe_args(**over):
    base = dict(
        project=None, files=["does-not-exist"], only=None, model=None,
        provider=None, pdf_converter=None, html_converter=None,
        verbose=False, timings=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("no active project"),
        ValueError("Unknown pdf_converter 'bogus'"),
        FileNotFoundError("missing path"),
    ],
)
def test_scribe_expected_errors_exit_one_on_stderr(monkeypatch, exc):
    """RuntimeError / ValueError / FileNotFoundError out of scribe become a
    one-line stderr message + exit 1, not an uncaught traceback."""
    from bartleby.lib import console

    errors: list[str] = []
    monkeypatch.setattr(console, "error", lambda m: errors.append(m))
    monkeypatch.setattr(console, "splash", lambda: None)

    def _boom(**kwargs):
        raise exc

    monkeypatch.setattr("bartleby.commands.scribe.main", _boom)

    with pytest.raises(SystemExit) as e:
        cli._scribe(_scribe_args())
    assert e.value.code == 1
    assert errors == [str(exc)]


# ---- project upgrade: bad name exits 1 on stderr (mirrors create) -----------


def test_project_upgrade_bad_name_exits_one(monkeypatch):
    errors = _capture_error(monkeypatch, project_cmd)
    with pytest.raises(SystemExit) as e:
        project_cmd.upgrade(name="../etc/passwd")
    assert e.value.code == 1
    assert len(errors) == 1


# ---- logs --limit must be a positive integer --------------------------------


@pytest.mark.parametrize("bad_limit", [-5, 0])
def test_logs_rejects_non_positive_limit(project, monkeypatch, bad_limit):
    errors = _capture_error(monkeypatch, logs_cmd)
    with pytest.raises(SystemExit) as e:
        logs_cmd.main(session=None, limit=bad_limit, project=project)
    assert e.value.code == 1
    assert len(errors) == 1
    assert "limit" in errors[0].lower()


def test_logs_unknown_session_routes_error_to_stderr(project, monkeypatch):
    """The 'no such session' error goes through console.error (stderr), so a
    `2>/dev/null` user sees nothing on stdout."""
    errors = _capture_error(monkeypatch, logs_cmd)
    with pytest.raises(SystemExit) as e:
        logs_cmd.main(session="nope", limit=50, project=project)
    assert e.value.code == 1
    assert len(errors) == 1


# ---- --provider/--model ignored when summaries are off ----------------------


def test_llm_override_ignored_warns(monkeypatch):
    warns: list[str] = []
    monkeypatch.setattr(resolve.console, "warn", lambda m: warns.append(m))
    prov, model = resolve._resolve_llm_provider(
        {"summary_depth": "none"},
        provider_override="anthropic", model_override=None,
    )
    assert (prov, model) == (None, None)
    assert len(warns) == 1
    assert "ignoring --provider/--model" in warns[0]


def test_llm_no_override_no_warn(monkeypatch):
    warns: list[str] = []
    monkeypatch.setattr(resolve.console, "warn", lambda m: warns.append(m))
    prov, model = resolve._resolve_llm_provider(
        {"summary_depth": "none"},
        provider_override=None, model_override=None,
    )
    assert (prov, model) == (None, None)
    assert warns == []


# ---- wizard bounded-int prompt: ocr_min_confidence accepts 0, rejects >100 ---


def _scripted_int(monkeypatch, values):
    queue = list(values)
    monkeypatch.setattr(
        "rich.prompt.IntPrompt.ask", lambda *a, **k: int(queue.pop(0))
    )


def test_bounded_int_accepts_zero(monkeypatch):
    _scripted_int(monkeypatch, [0])
    assert config_cmd._prompt_bounded_int(
        "x", 30, min_value=0, max_value=100, help_text=""
    ) == 0


@pytest.mark.parametrize("bad", [101, 250, -1])
def test_bounded_int_rejects_out_of_range(monkeypatch, bad):
    # First a rejected value, then an in-range one to break the loop.
    _scripted_int(monkeypatch, [bad, 50])
    assert config_cmd._prompt_bounded_int(
        "x", 30, min_value=0, max_value=100, help_text=""
    ) == 50


# ---- wsjpt key prompt advertises GEMINI_API_KEY, not WSJPT_API_KEY -----------


def test_wsjpt_api_key_help_names_gemini():
    help_text = config_cmd._api_key_help("wsjpt")
    assert "GEMINI_API_KEY" in help_text
    assert "WSJPT_API_KEY" not in help_text


def test_non_wsjpt_api_key_help_names_provider_var():
    assert "ANTHROPIC_API_KEY" in config_cmd._api_key_help("anthropic")
    assert "OPENAI_API_KEY" in config_cmd._api_key_help("openai")
