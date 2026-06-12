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
from bartleby.commands import serve as serve_cmd
from bartleby.commands import session as session_cmd
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


# ---- bad --project name exits 1 on stderr at every resolve site (#488/#491) --
#
# resolve_project_name / set_active_project now raise ValueError on a traversal
# name; logs / session / project-use must route that through console.error +
# exit 1 like their RuntimeError/FileNotFoundError siblings, never a traceback.


def test_logs_bad_project_name_exits_one(monkeypatch):
    errors = _capture_error(monkeypatch, logs_cmd)
    with pytest.raises(SystemExit) as e:
        logs_cmd.main(session=None, limit=50, project="../../x")
    assert e.value.code == 1
    assert len(errors) == 1
    assert "Invalid project name" in errors[0]


def test_session_current_bad_project_name_exits_one(monkeypatch):
    errors = _capture_error(monkeypatch, session_cmd)
    with pytest.raises(SystemExit) as e:
        session_cmd.current(project="../x")
    assert e.value.code == 1
    assert len(errors) == 1
    assert "Invalid project name" in errors[0]


def test_project_use_bad_project_name_exits_one(monkeypatch):
    errors = _capture_error(monkeypatch, project_cmd)
    with pytest.raises(SystemExit) as e:
        project_cmd.use(name="../x")
    assert e.value.code == 1
    assert len(errors) == 1
    assert "Invalid project name" in errors[0]


# ---- serve --project routes through name validation before any path build ----


def test_serve_bad_project_name_exits_one_before_path_build(monkeypatch):
    """``serve --project '../x'`` must be rejected by validate_project_name at
    the top of _override_project, before project_db_path is ever called."""
    errors = _capture_error(monkeypatch, serve_cmd)

    def _no_path(name):  # pragma: no cover - must not be reached
        raise AssertionError("project_db_path called before name validation")

    monkeypatch.setattr(serve_cmd, "project_db_path", _no_path)
    with pytest.raises(SystemExit) as e:
        serve_cmd._override_project("../x")
    assert e.value.code == 1
    assert len(errors) == 1
    assert "Invalid project name" in errors[0]


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


# ---- project import overwrite: prompt unless --yes, like project delete (#528) -


def _stub_import(monkeypatch):
    """Replace the S3-touching import_project with a recorder.

    Returns the call log; each entry is the ``force`` value the handler passed.
    The handler imports ``import_project`` at call time from
    ``bartleby.share.import_``, so patching it there is enough.
    """
    calls: list[bool] = []

    def fake(name, from_url, *, client=None, without_tags=False, force=False):
        calls.append(force)
        return {"project": name, "source": from_url, "file_count": 2,
                "tags_dropped": without_tags}

    monkeypatch.setattr("bartleby.share.import_.import_project", fake)
    return calls


def _arm_confirm(monkeypatch, answer):
    """Stub the interactive confirm; return the list of prompts it received."""
    asked: list[str] = []

    def fake_ask(prompt, *a, **k):
        asked.append(prompt)
        return answer

    monkeypatch.setattr(project_cmd.Confirm, "ask", staticmethod(fake_ask))
    return asked


def _do_import(name, yes):
    project_cmd.import_(name=name, from_url="s3://bucket/corpora/pub",
                        without_tags=False, yes=yes)


def test_import_yes_overwrites_existing_without_prompt(project, monkeypatch):
    calls = _stub_import(monkeypatch)
    asked = _arm_confirm(monkeypatch, True)  # would say yes, but must not be asked
    _do_import("alpha", yes=True)  # "alpha" exists (project fixture)
    assert asked == []  # --yes skips the prompt entirely
    assert calls == [True]  # ...and overwrites


def test_import_existing_prompts_and_overwrites_on_confirm(project, monkeypatch):
    calls = _stub_import(monkeypatch)
    asked = _arm_confirm(monkeypatch, True)
    _do_import("alpha", yes=False)
    assert len(asked) == 1  # the same-name overwrite is confirmed
    assert calls == [True]


def test_import_existing_cancels_on_decline(project, monkeypatch):
    calls = _stub_import(monkeypatch)
    asked = _arm_confirm(monkeypatch, False)
    _do_import("alpha", yes=False)
    assert len(asked) == 1
    assert calls == []  # declined -> import_project never runs


def test_import_fresh_name_needs_no_prompt(monkeypatch):
    # No project created: a non-colliding name imports straight through.
    calls = _stub_import(monkeypatch)
    asked = _arm_confirm(monkeypatch, True)
    _do_import("brandnew", yes=False)
    assert asked == []  # nothing to overwrite -> no confirm
    assert calls == [False]
