"""Provider unit tests — mock the SDKs, exercise summarize + analyze_image.

We never call out to a real provider in CI. Each test patches the underlying
SDK client with a fake so we can assert the wire shape (tool_use vs.
response_format vs. format=) and the validation behavior.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bartleby.providers.base import DocumentSummary, ImageAnalysis


_SUMMARY_INPUT = {
    "title": "T", "description": "D", "text": "Some summary text.",
}
_IMAGE_INPUT = {
    "kind": "scene", "text": "WELCOME", "description": "A cat sitting.", "notes": "",
}


# ---------- Anthropic ----------


class _FakeAnthropicResponse:
    def __init__(self, blocks):
        self.content = blocks


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


def test_anthropic_analyze_image_validates_tool_input(monkeypatch):
    response = _FakeAnthropicResponse([
        _block("tool_use", name="save_image_analysis", input_=_IMAGE_INPUT),
    ])
    fake = _install_anthropic(monkeypatch, response)

    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    result = p.analyze_image(b"\xff\xd8\xff", model="claude-haiku-4-5")

    assert isinstance(result, ImageAnalysis)
    assert result.kind == "scene"
    assert result.text == "WELCOME"
    # Image is passed as a base64 block.
    content = fake.last_call["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in content)
    assert fake.last_call["tool_choice"]["name"] == "save_image_analysis"


def test_anthropic_missing_tool_use_raises(monkeypatch):
    response = _FakeAnthropicResponse([_block("text")])
    _install_anthropic(monkeypatch, response)
    from bartleby.providers.anthropic import AnthropicProvider
    p = AnthropicProvider()
    with pytest.raises(RuntimeError, match="did not include"):
        p.analyze_image(b"\x00", model="m")


def test_anthropic_invalid_tool_input_raises(monkeypatch):
    bad = {"kind": "scene"}  # missing text/description/notes
    response = _FakeAnthropicResponse([
        _block("tool_use", name="save_image_analysis", input_=bad),
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


def test_openai_analyze_image_returns_parsed_pydantic(monkeypatch):
    parsed = ImageAnalysis(**_IMAGE_INPUT)
    fake = _install_openai(monkeypatch, _FakeOpenAIResponse(parsed=parsed))
    from bartleby.providers.openai import OpenAIProvider
    p = OpenAIProvider()
    result = p.analyze_image(b"\xff\xd8\xff", model="gpt-5-mini")
    assert result is parsed
    assert fake.last_call["response_format"] is ImageAnalysis
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
                           _FakeOllamaResponse(content=json.dumps(_IMAGE_INPUT)))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    result = p.analyze_image(b"\xff\xd8\xff", model="qwen2.5-vl:7b")
    assert isinstance(result, ImageAnalysis)
    assert result.text == "WELCOME"
    msg = fake.last_call["messages"][0]
    assert msg["images"] == [b"\xff\xd8\xff"]
    assert fake.last_call["format"] == ImageAnalysis.model_json_schema()


def test_ollama_empty_content_raises(monkeypatch):
    _install_ollama(monkeypatch, _FakeOllamaResponse(content=""))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    with pytest.raises(RuntimeError, match="empty response"):
        p.analyze_image(b"\x00", model="m")


def test_ollama_malformed_json_raises(monkeypatch):
    _install_ollama(monkeypatch, _FakeOllamaResponse(content='{"kind": "scene"}'))
    from bartleby.providers.ollama import OllamaProvider
    p = OllamaProvider(base_url="http://test:11434")
    with pytest.raises(RuntimeError, match="failed .* validation"):
        p.analyze_image(b"\x00", model="m")
