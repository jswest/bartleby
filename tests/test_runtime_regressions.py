"""Regression tests for real past bugs in providers, `bartleby ready`, and
BARTLEBY_HOME. Each section names the issue it guards.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from bartleby import config
from bartleby.providers.base import DocumentSummary, VlmDescription
import bartleby.commands.ready as ready


# ---------- providers: #504 Anthropic temperature/effort deny-list ----------


_SUMMARY_INPUT = {
    "title": "T", "description": "D", "text": "Some summary text.",
}


class _FakeAnthropicResponse:
    def __init__(self, blocks, stop_reason="tool_use"):
        self.content = blocks
        self.stop_reason = stop_reason


def _block(type_, name=None, input_=None):
    b = SimpleNamespace(type=type_, name=name, input=input_)
    return b


class _FakeAnthropicClient:
    def __init__(self, response):
        self._response = response
        self.last_call: dict | None = None

    def messages_create(self, **kwargs):
        self.last_call = kwargs
        return self._response


def _install_anthropic(monkeypatch, response):
    from bartleby.providers import anthropic as mod
    fake = _FakeAnthropicClient(response)
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake.messages_create))
    monkeypatch.setattr(mod, "Anthropic", lambda: fake_client)
    return fake


def _summarize_with_effort(monkeypatch, *, model, effort):
    response = _FakeAnthropicResponse([
        _block("tool_use", name="save_summary", input_=_SUMMARY_INPUT),
    ])
    fake = _install_anthropic(monkeypatch, response)
    from bartleby.providers.anthropic import AnthropicProvider
    AnthropicProvider().summarize(
        "a doc", model=model, temperature=0.0, reasoning_effort=effort,
    )
    return fake


def test_anthropic_current_model_drops_temperature_and_uses_effort(monkeypatch):
    # Deny-list inversion: a current model not on either older deny-list
    # (claude-fable-5) must default SAFE — no temperature sent (it would 400, the
    # #250 regression), effort applied. An allowlist that predates this model
    # would have done the opposite.
    fake = _summarize_with_effort(monkeypatch, model="claude-fable-5", effort="high")
    assert "temperature" not in fake.last_call
    assert fake.last_call["output_config"] == {"effort": "high"}


def test_anthropic_future_model_defaults_safe(monkeypatch):
    # Any model released after this code (here a stand-in unknown id) is off both
    # deny-lists, so it inherits the safe path: temperature dropped, effort sent.
    fake = _summarize_with_effort(monkeypatch, model="claude-opus-5-0", effort="medium")
    assert "temperature" not in fake.last_call
    assert fake.last_call["output_config"] == {"effort": "medium"}


def test_anthropic_denylisted_older_model_keeps_temperature_no_effort(monkeypatch):
    # The pre-effort, temperature-accepting deny-list (Sonnet 4.5 here) must keep
    # its existing behavior exactly: temperature forwarded, no effort knob sent.
    fake = _summarize_with_effort(monkeypatch, model="claude-sonnet-4-5", effort="high")
    assert fake.last_call["temperature"] == 0.0
    assert "output_config" not in fake.last_call


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-1", "claude-opus-4-0", "claude-sonnet-4-0", "claude-3-haiku",
        # Dated snapshot IDs must be covered too — the alias prefixes don't match them.
        "claude-opus-4-20250514", "claude-sonnet-4-20250514",
    ],
)
def test_anthropic_pre_4_5_models_keep_temperature_and_skip_effort(monkeypatch, model):
    # The deny-lists say "Sonnet 4.5 / Haiku 4.5 AND EARLIER" — still-reachable
    # pre-4.5 models (Opus 4.0/4.1, Sonnet 4.0, claude-3, and their dated
    # snapshot IDs) also 400 on output_config.effort and still accept
    # temperature. They must inherit the OLD behavior: temperature forwarded for
    # determinism, effort never sent (sending it would 400 every summarize call).
    fake = _summarize_with_effort(monkeypatch, model=model, effort="high")
    assert fake.last_call["temperature"] == 0.0
    assert "output_config" not in fake.last_call


# ---------- wsjpt: #755 one Jpt built per call leaked sockets ----------


class _FakeWsjptJpt:
    """Stand-in for wsjpt.Jpt: counts constructions, does no I/O."""

    construct_count = 0

    def __init__(self, pydantic_model, model_config=None, custom_instructions=None):
        _FakeWsjptJpt.construct_count += 1
        self._pydantic_model = pydantic_model

    def parse(self, input_text=None, binary_files=None):
        if self._pydantic_model is DocumentSummary:
            return DocumentSummary(title="T", description="D", text="Body")
        return VlmDescription(description="desc", notes="")


def _install_fake_wsjpt(monkeypatch):
    _FakeWsjptJpt.construct_count = 0
    fake_module = SimpleNamespace(Jpt=_FakeWsjptJpt, ModelConfig=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "wsjpt", fake_module)


def test_wsjpt_jpt_built_once_and_reused_across_calls(monkeypatch):
    # #755: WsjptProvider built a fresh Jpt (and its genai.Client, i.e. open
    # sockets) on every summarize/classify/analyze_image call. Thousands of
    # caption calls in one scribe run piled up past the fd limit until the
    # process hit EMFILE. A Jpt for a given (schema, model, instructions) must
    # be built once and reused.
    _install_fake_wsjpt(monkeypatch)
    from bartleby.providers.wsjpt import WsjptProvider

    provider = WsjptProvider()
    for _ in range(5):
        provider.summarize("doc text", model="fast", temperature=0.0)
    for _ in range(5):
        provider.analyze_image(b"bytes", model="fast", temperature=0.0)

    assert _FakeWsjptJpt.construct_count == 2  # one per (schema, model, instructions)


def test_wsjpt_jpt_shared_safely_across_threads(monkeypatch):
    # Caption and summarize both run calls through a 4-thread pool; concurrent
    # calls for the same key must not race into building more than one Jpt.
    _install_fake_wsjpt(monkeypatch)
    from bartleby.providers.wsjpt import WsjptProvider

    provider = WsjptProvider()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(
            lambda _: provider.summarize("doc text", model="fast", temperature=0.0),
            range(20),
        ))

    assert _FakeWsjptJpt.construct_count == 1


# ---------- ready: #206 stale ready-marker ----------


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "skills" / "bartleby"


def _marker(dest):
    return json.loads((dest / ready.MARKER_NAME).read_text())


def test_stale_marker_refreshes_without_reinstall(dest, monkeypatch):
    ready.main(dest=dest)
    # Simulate a content-identical version bump: the skill files are byte-for-byte
    # the same, but the marker was written by an older bartleby.
    marker = _marker(dest)
    (dest / ready.MARKER_NAME).write_text(
        json.dumps({"version": "0.0.1", "hash": marker["hash"]}) + "\n"
    )

    calls = []
    monkeypatch.setattr(ready, "_install", lambda *a, **k: calls.append(1))
    ready.main(dest=dest)

    assert calls == []  # marker-only refresh, never a full reinstall
    assert _marker(dest)["version"] == ready.__version__


def test_check_agrees_with_default_after_refresh(dest, monkeypatch):
    ready.main(dest=dest)
    marker = _marker(dest)
    (dest / ready.MARKER_NAME).write_text(
        json.dumps({"version": "0.0.1", "hash": marker["hash"]}) + "\n"
    )

    messages = []
    for name in ("complete", "big", "warn"):
        monkeypatch.setattr(ready.console, name, lambda m, _m=messages: _m.append(m))

    ready.main(dest=dest)  # default path heals the stamp to the running version
    assert messages == [
        f"Updated skill marker v0.0.1 → v{ready.__version__} at {dest}"
    ]

    messages.clear()
    ready.main(dest=dest, check=True)  # read-only; now reports the healed version

    assert messages == [f"Skill is up to date (v{ready.__version__}) at {dest}."]


# ---------- #393 BARTLEBY_HOME override ----------


def test_override_relocates_whole_tree(tmp_path, monkeypatch):
    home = tmp_path / "sandbox"
    monkeypatch.setenv("BARTLEBY_HOME", str(home))
    assert config.bartleby_dir() == home
    assert config.projects_dir() == home / "projects"
    assert config.config_path() == home / "config.yaml"
    assert config.scratch_dir() == home / "tmp"


def test_resolved_lazily_after_env_set(tmp_path, monkeypatch):
    # The crux of GH-0393: read at call time, so a change *after* import takes
    # effect. A module-level constant would have frozen the path at import and
    # silently ignored both of these.
    monkeypatch.setenv("BARTLEBY_HOME", str(tmp_path / "a"))
    assert config.bartleby_dir() == tmp_path / "a"
    monkeypatch.setenv("BARTLEBY_HOME", str(tmp_path / "b"))
    assert config.bartleby_dir() == tmp_path / "b"


def test_override_expands_user(monkeypatch):
    monkeypatch.setenv("BARTLEBY_HOME", "~/some-bartleby-sandbox")
    assert config.bartleby_dir() == Path.home() / "some-bartleby-sandbox"


def test_falls_back_to_home_when_unset(monkeypatch):
    monkeypatch.delenv("BARTLEBY_HOME", raising=False)
    # Read-only assertion: resolve the path, never create anything under it.
    assert config.bartleby_dir() == Path.home() / ".bartleby"
