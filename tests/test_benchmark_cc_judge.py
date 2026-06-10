import json
from types import SimpleNamespace

import pytest

from bartleby.benchmark import cc_judge, summarize
from bartleby.benchmark.refs import LOCAL_PROVIDERS, PROVIDERS, ModelRef

SRC = "Rayleigh scattering makes the sky blue."
SUMMARY = {"title": "Sky", "description": "Why the sky is blue.",
           "text": "Short wavelengths scatter more."}
GOOD = {"faithfulness": 5, "coverage": 4, "conciseness": 5,
        "constraint_compliance": 5, "rationale": "ok"}


def _runner(returncode=0, stdout="", stderr=""):
    captured = {}

    def run(argv, *, env):
        captured["argv"] = argv
        captured["env"] = env
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    run.captured = captured
    return run


def _envelope(**kw):
    base = {"subtype": "success", "is_error": False, "structured_output": GOOD}
    base.update(kw)
    return json.dumps(base)


def test_anthropic_cc_is_a_local_provider():
    assert "anthropic-cc" in PROVIDERS
    assert "anthropic-cc" in LOCAL_PROVIDERS


def test_resolve_model_aliases_and_passthrough():
    assert cc_judge.resolve_model("fable5") == "claude-fable-5"
    assert cc_judge.resolve_model("opus4.8") == "claude-opus-4-8"
    assert cc_judge.resolve_model("claude-opus-4-8") == "claude-opus-4-8"
    assert cc_judge.resolve_model("opus") == "opus"


def test_subscription_env_strips_billing_override_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    env, stripped = cc_judge._subscription_env()
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert set(stripped) == {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}


def test_judge_schema_uses_enum_axes_not_bounds():
    schema = cc_judge._judge_schema()
    assert schema["additionalProperties"] is False
    for axis in ("faithfulness", "coverage", "conciseness", "constraint_compliance"):
        prop = schema["properties"][axis]
        assert prop["enum"] == [1, 2, 3, 4, 5]
        assert "minimum" not in prop and "maximum" not in prop


def test_judge_summary_cc_success_and_argv(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    runner = _runner(stdout=_envelope())
    result = cc_judge.judge_summary_cc("opus4.8", SRC, SUMMARY, runner=runner)
    assert result["ok"] is True
    assert result["scores"]["mean"] == pytest.approx(4.75)

    argv = runner.captured["argv"]
    assert argv[:2] == ["claude", "-p"]
    assert "--output-format" in argv and "json" in argv
    assert "--json-schema" in argv
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"  # alias resolved
    assert "--bare" not in argv
    assert argv[argv.index("--tools") + 1] == ""  # tools disabled
    # auth safety: the api key never reaches the subprocess environment
    assert "ANTHROPIC_API_KEY" not in runner.captured["env"]
    # blind: the prompt carries the source but no model identity
    assert SRC in argv[2]


def test_judge_summary_cc_nonzero_exit_is_a_failure_record():
    runner = _runner(returncode=2, stderr="not logged in")
    result = cc_judge.judge_summary_cc("opus", SRC, SUMMARY, runner=runner)
    assert result["ok"] is False
    assert "exited 2" in result["error"] and "not logged in" in result["error"]


def test_judge_summary_cc_retry_exhausted_is_a_failure_record():
    runner = _runner(stdout=_envelope(
        subtype="error_max_structured_output_retries", is_error=True))
    result = cc_judge.judge_summary_cc("opus", SRC, SUMMARY, runner=runner)
    assert result["ok"] is False
    assert "error_max_structured_output_retries" in result["error"]


def test_judge_summary_cc_unparseable_output_is_a_failure_record():
    result = cc_judge.judge_summary_cc(
        "opus", SRC, SUMMARY, runner=_runner(stdout="not json"))
    assert result["ok"] is False
    assert "unparseable" in result["error"]


def test_judge_summary_cc_out_of_range_axis_fails_pydantic_check():
    bad = dict(GOOD, faithfulness=9)  # outside 1-5 — Pydantic ge/le must catch
    runner = _runner(stdout=_envelope(structured_output=bad))
    result = cc_judge.judge_summary_cc("opus", SRC, SUMMARY, runner=runner)
    assert result["ok"] is False
    assert "schema validation" in result["error"]


def test_preflight_missing_cli_aborts(monkeypatch):
    monkeypatch.setattr(cc_judge.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="Claude Code CLI on PATH"):
        cc_judge.preflight()


def test_preflight_not_logged_in_aborts(monkeypatch):
    monkeypatch.setattr(cc_judge.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(cc_judge.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='{"loggedIn": false}', stderr=""))
    with pytest.raises(SystemExit, match="requires a logged-in Claude subscription"):
        cc_judge.preflight()


def test_preflight_ok_warns_when_key_present(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setattr(cc_judge.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(cc_judge.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='{"loggedIn": true}', stderr=""))
    cc_judge.preflight()  # no raise
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_summarize_rejects_anthropic_cc(tmp_path):
    (tmp_path / "corpus.yaml").write_text("doc-a: a.pdf\n")
    from bartleby.benchmark.stores import BenchmarkRoot
    with pytest.raises(SystemExit, match="judge-only"):
        summarize.run(BenchmarkRoot(tmp_path),
                      models=[ModelRef("anthropic-cc", "claude-opus-4-8")],
                      documents=None)
