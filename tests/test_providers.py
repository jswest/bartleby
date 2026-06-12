"""Provider unit tests — mock the SDKs, exercise summarize + analyze_image.

We never call out to a real provider in CI. Each test patches the underlying
SDK client with a fake so we can assert the wire shape (tool_use vs.
response_format vs. format=) and the validation behavior.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from bartleby.providers.base import DocumentSummary, VlmDescription


_SUMMARY_INPUT = {
    "title": "T", "description": "D", "text": "Some summary text.",
}
_VLM_INPUT = {"description": "A cat sitting.", "notes": ""}


# ---------- Anthropic ----------


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


def test_anthropic_summarize_validates_tool_input(monkeypatch):
    response = _FakeAnthropicResponse([
        _block("text"),
        _block("tool_use", name="save_summary", input_=_SUMMARY_INPUT),
    ])
    fake = _install_anthropic(monkeypatch, response)

    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    result = p.summarize("a doc", model="claude-haiku-4-5", temperature=0.0)

    assert isinstance(result, DocumentSummary)
    assert result.title == "T"
    # The summary call must force the save_summary tool.
    assert fake.last_call["tool_choice"]["name"] == "save_summary"
    # No reasoning_effort + a non-effort model → plain temperature, no effort knob.
    assert "output_config" not in fake.last_call
    assert fake.last_call["temperature"] == 0.0


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


def test_anthropic_effort_sent_for_capable_model_drops_temperature(monkeypatch):
    # Opus 4.8 accepts output_config.effort and rejects temperature → effort set,
    # temperature omitted.
    fake = _summarize_with_effort(monkeypatch, model="claude-opus-4-8", effort="low")
    assert fake.last_call["output_config"] == {"effort": "low"}
    assert "temperature" not in fake.last_call


def test_anthropic_effort_keeps_temperature_on_older_capable_model(monkeypatch):
    # Opus 4.6 supports effort but still accepts temperature → both sent.
    fake = _summarize_with_effort(monkeypatch, model="claude-opus-4-6", effort="medium")
    assert fake.last_call["output_config"] == {"effort": "medium"}
    assert fake.last_call["temperature"] == 0.0


def test_anthropic_out_of_enum_effort_raises_named_valueerror(monkeypatch):
    # Backstop for the scribe-level gate (#489): a caller reaching the provider
    # directly with an out-of-enum effort gets a ValueError naming the field —
    # not the bare KeyError a raw _EFFORT_MAP[...] lookup would raise.
    with pytest.raises(ValueError) as exc:
        _summarize_with_effort(monkeypatch, model="claude-opus-4-8", effort="xhigh")
    msg = str(exc.value)
    assert "reasoning_effort" in msg
    assert "xhigh" in msg


def test_anthropic_effort_skipped_for_uncapable_model(monkeypatch):
    # Haiku 4.5 would 400 on output_config.effort → never sent; temperature stays.
    fake = _summarize_with_effort(monkeypatch, model="claude-haiku-4-5", effort="high")
    assert "output_config" not in fake.last_call
    assert fake.last_call["temperature"] == 0.0


def test_anthropic_drops_temperature_on_47plus_without_effort(monkeypatch):
    # Opus 4.7+ rejects temperature whether or not effort is configured — so an
    # unset reasoning_effort must still drop it, or every summarize call 400s.
    fake = _summarize_with_effort(monkeypatch, model="claude-opus-4-8", effort=None)
    assert "output_config" not in fake.last_call
    assert "temperature" not in fake.last_call


def test_anthropic_minimal_effort_maps_to_low(monkeypatch):
    # Anthropic has no "minimal" — our enum's minimal maps to the lowest, "low".
    fake = _summarize_with_effort(monkeypatch, model="claude-opus-4-8", effort="minimal")
    assert fake.last_call["output_config"] == {"effort": "low"}


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


def _classify_on(monkeypatch, model):
    class _Schema(BaseModel):
        ok: bool

    response = _FakeAnthropicResponse([
        _block("tool_use", name="save_classification", input_={"ok": True}),
    ])
    fake = _install_anthropic(monkeypatch, response)
    from bartleby.providers.anthropic import AnthropicProvider
    result = AnthropicProvider().classify(
        "prompt", model=model, schema=_Schema, temperature=0.3,
    )
    return fake, result


def test_anthropic_classify_drops_temperature_on_47plus(monkeypatch):
    # Opus 4.7+ 400s on temperature; classify reuses the summarizer's model, so it
    # must drop temperature too or every tag-classification call fails.
    fake, result = _classify_on(monkeypatch, "claude-opus-4-8")
    assert result.ok is True
    assert fake.last_call["tool_choice"]["name"] == "save_classification"
    assert "temperature" not in fake.last_call


def test_anthropic_classify_keeps_temperature_on_older_model(monkeypatch):
    # Models that accept temperature still get it — don't over-drop.
    fake, _ = _classify_on(monkeypatch, "claude-haiku-4-5")
    assert fake.last_call["temperature"] == 0.3


def test_anthropic_analyze_image_validates_tool_input(monkeypatch):
    response = _FakeAnthropicResponse([
        _block("tool_use", name="save_image_description", input_=_VLM_INPUT),
    ])
    fake = _install_anthropic(monkeypatch, response)

    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    result = p.analyze_image(b"\xff\xd8\xff", model="claude-haiku-4-5")

    assert isinstance(result, VlmDescription)
    assert result.description == "A cat sitting."
    # Image is passed as a base64 block.
    content = fake.last_call["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in content)
    assert fake.last_call["tool_choice"]["name"] == "save_image_description"


def test_anthropic_missing_tool_use_raises(monkeypatch):
    response = _FakeAnthropicResponse([_block("text")])
    _install_anthropic(monkeypatch, response)
    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    with pytest.raises(RuntimeError, match="did not include"):
        p.analyze_image(b"\x00", model="m")


def test_anthropic_max_tokens_truncation_raises_named_error(monkeypatch):
    # A max_tokens truncation cuts the response off before the forced tool block
    # lands, so the tool call is missing. The error must NAME the max_tokens
    # truncation rather than the opaque "did not include" message — the fix
    # (raise max_tokens) is different from a genuinely tool-less response.
    response = _FakeAnthropicResponse([_block("text")], stop_reason="max_tokens")
    _install_anthropic(monkeypatch, response)
    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    with pytest.raises(RuntimeError, match="max_tokens") as exc:
        p.analyze_image(b"\x00", model="m")
    assert "did not include" not in str(exc.value)


def test_anthropic_invalid_tool_input_raises(monkeypatch):
    bad = {"notes": "x"}  # missing description
    response = _FakeAnthropicResponse([
        _block("tool_use", name="save_image_description", input_=bad),
    ])
    _install_anthropic(monkeypatch, response)
    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    with pytest.raises(RuntimeError, match="failed schema validation"):
        p.analyze_image(b"\x00", model="m")


# ---------- OpenAI ----------


class _FakeOpenAIChoice:
    def __init__(self, parsed=None, refusal=None):
        self.message = SimpleNamespace(parsed=parsed, refusal=refusal)


class _FakeOpenAIResponse:
    def __init__(self, parsed=None, refusal=None):
        self.choices = [_FakeOpenAIChoice(parsed=parsed, refusal=refusal)]


class _FakeOpenAIClient:
    def __init__(self, response):
        self._response = response
        self.last_call: dict | None = None

    def _parse(self, **kwargs):
        self.last_call = kwargs
        return self._response


def _install_openai(monkeypatch, response):
    from bartleby.providers import openai as mod
    fake = _FakeOpenAIClient(response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=fake._parse))
    )
    monkeypatch.setattr(mod, "OpenAI", lambda: fake_client)
    return fake


def test_openai_summarize_returns_parsed_pydantic(monkeypatch):
    parsed = DocumentSummary(**_SUMMARY_INPUT)
    fake = _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))
    from bartleby.providers.openai import OpenAIProvider
    p = OpenAIProvider()
    result = p.summarize("doc", model="gpt-5-mini", temperature=0.0)
    assert result is parsed
    assert fake.last_call["response_format"] is DocumentSummary
    # GPT-5 models reject any non-default temperature, so we never send it.
    assert "temperature" not in fake.last_call
    # No reasoning_effort configured → the parameter is omitted entirely.
    assert "reasoning_effort" not in fake.last_call


def test_openai_summarize_forwards_reasoning_effort(monkeypatch):
    parsed = DocumentSummary(**_SUMMARY_INPUT)
    fake = _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))
    from bartleby.providers.openai import OpenAIProvider
    OpenAIProvider().summarize(
        "doc", model="gpt-5-nano", temperature=0.0, reasoning_effort="minimal",
    )
    # gpt-5 takes the effort string verbatim (minimal | low | medium | high).
    assert fake.last_call["reasoning_effort"] == "minimal"


def test_openai_classify_omits_temperature(monkeypatch):
    class _Schema(BaseModel):
        ok: bool

    parsed = _Schema(ok=True)
    fake = _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))
    from bartleby.providers.openai import OpenAIProvider
    p = OpenAIProvider()
    result = p.classify("prompt", model="gpt-5-nano", schema=_Schema, temperature=0.3)
    assert result is parsed
    assert "temperature" not in fake.last_call


def test_openai_warns_once_on_dropped_temperature(monkeypatch):
    from bartleby.providers import openai as mod
    monkeypatch.setattr(mod, "_temperature_warned", False)
    warnings: list[str] = []
    monkeypatch.setattr(mod.console, "warn", warnings.append)
    parsed = DocumentSummary(**_SUMMARY_INPUT)
    _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))

    p = mod.OpenAIProvider()
    p.summarize("doc", model="gpt-5-mini", temperature=0.0)
    p.summarize("doc", model="gpt-5-mini", temperature=0.0)

    # Warned about the dropped value, exactly once across both calls.
    assert len(warnings) == 1
    assert "temperature=0.0" in warnings[0]


def test_openai_no_warning_when_temperature_is_default(monkeypatch):
    from bartleby.providers import openai as mod
    monkeypatch.setattr(mod, "_temperature_warned", False)
    warnings: list[str] = []
    monkeypatch.setattr(mod.console, "warn", warnings.append)
    parsed = DocumentSummary(**_SUMMARY_INPUT)
    _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))

    mod.OpenAIProvider().summarize("doc", model="gpt-5-mini", temperature=1.0)

    # 1.0 is the API default — nothing is dropped, so no warning.
    assert warnings == []


def test_openai_analyze_image_returns_parsed_pydantic(monkeypatch):
    parsed = VlmDescription(**_VLM_INPUT)
    fake = _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))
    from bartleby.providers.openai import OpenAIProvider
    p = OpenAIProvider()
    result = p.analyze_image(b"\xff\xd8\xff", model="gpt-5-mini")
    assert result is parsed
    assert fake.last_call["response_format"] is VlmDescription
    # Image is passed as a data URL in the content blocks.
    content = fake.last_call["messages"][0]["content"]
    image_blocks = [b for b in content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_refusal_raises(monkeypatch):
    _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=None, refusal="nope"))
    from bartleby.providers.openai import OpenAIProvider
    p = OpenAIProvider()
    with pytest.raises(RuntimeError, match="refusal='nope'"):
        p.analyze_image(b"\x00", model="m")


# ---------- Ollama ----------


class _FakeOllamaResponse:
    def __init__(self, content):
        self.message = SimpleNamespace(content=content)


class _FakeOllamaClient:
    def __init__(self, response):
        self._response = response
        self.last_call: dict | None = None

    def chat(self, **kwargs):
        self.last_call = kwargs
        return self._response


def _install_ollama(monkeypatch, response):
    from bartleby.providers import ollama as mod
    fake = _FakeOllamaClient(response)
    monkeypatch.setattr(mod.ollama, "Client", lambda host: fake)
    return fake


def test_ollama_summarize_validates_json(monkeypatch):
    fake = _install_ollama(monkeypatch,
                           _FakeOllamaResponse(content=json.dumps(_SUMMARY_INPUT)))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    result = p.summarize("doc", model="gpt-oss:20b", temperature=0.0)
    assert isinstance(result, DocumentSummary)
    assert fake.last_call["format"] == DocumentSummary.model_json_schema()


def test_ollama_analyze_image_passes_bytes(monkeypatch):
    fake = _install_ollama(monkeypatch,
                           _FakeOllamaResponse(content=json.dumps(_VLM_INPUT)))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    result = p.analyze_image(b"\xff\xd8\xff", model="qwen2.5-vl:7b")
    assert isinstance(result, VlmDescription)
    assert result.description == "A cat sitting."
    msg = fake.last_call["messages"][0]
    assert msg["images"] == [b"\xff\xd8\xff"]
    assert fake.last_call["format"] == VlmDescription.model_json_schema()


def test_ollama_empty_content_raises(monkeypatch):
    _install_ollama(monkeypatch, _FakeOllamaResponse(content=""))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    with pytest.raises(RuntimeError, match="empty response"):
        p.analyze_image(b"\x00", model="m")


def test_ollama_malformed_json_raises(monkeypatch):
    _install_ollama(monkeypatch, _FakeOllamaResponse(content='{"notes": "x"}'))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    with pytest.raises(RuntimeError, match="failed .* validation"):
        p.analyze_image(b"\x00", model="m")


# ---------- wsjpt ----------
#
# wsjpt is a firewall-only optional install, absent from CI, so we inject a fake
# `wsjpt` module (Jpt + ModelConfig) and exercise the provider against it.


def _install_wsjpt(monkeypatch, parse_return):
    """Inject a fake `wsjpt` module; return the dict the fake Jpt records into."""
    calls: dict = {}

    class _FakeJpt:
        def __init__(self, schema, *, model_config, custom_instructions=None):
            calls["schema"] = schema
            calls["model_config"] = model_config
            calls["custom_instructions"] = custom_instructions

        def parse(self, *, input_text=None, binary_files=None):
            calls["input_text"] = input_text
            calls["binary_files"] = binary_files
            return parse_return

    fake_module = SimpleNamespace(
        Jpt=_FakeJpt,
        ModelConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "wsjpt", fake_module)
    return calls


def test_wsjpt_classify_routes_schema_and_prompt(monkeypatch):
    class _TagsAssignment(BaseModel):
        tag_ids: list[int]

    expected = _TagsAssignment(tag_ids=[1, 2])
    calls = _install_wsjpt(monkeypatch, expected)

    from bartleby.providers.wsjpt import WsjptProvider
    p = WsjptProvider()
    result = p.classify(
        "the self-contained prompt",
        model="fast",
        schema=_TagsAssignment,
        temperature=0.7,
    )

    # classify returns wsjpt's already-validated instance unchanged.
    assert result is expected
    # The caller's schema and prompt drive the parse; the prompt is
    # self-contained, so no custom_instructions are attached.
    assert calls["schema"] is _TagsAssignment
    assert calls["input_text"] == "the self-contained prompt"
    assert calls["custom_instructions"] is None
    assert calls["model_config"].model == "fast"
