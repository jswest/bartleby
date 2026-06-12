"""Anthropic provider — structured output via tool-use."""

from __future__ import annotations

import base64

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from bartleby.lib import console
from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    build_summary_messages,
)


_SUMMARY_TOOL = "save_summary"
_IMAGE_TOOL = "save_image_description"
_CLASSIFY_TOOL = "save_classification"

# Both knobs are deny-lists of the OLDER models, so a current model
# (claude-fable-5) and any future model take the safe path automatically — an
# allowlist would silently mis-handle every model released after this code,
# reintroducing the #250 400 and dropping effort.
#
# Effort (output_config.effort) is GA on Opus 4.5+, Sonnet 4.6, Fable 5, and
# every later model; it 400s only on Sonnet 4.5 / Haiku 4.5 and earlier. So we
# deny-list the pre-effort models and send effort everywhere else. We deliberately
# do NOT pair it with a thinking block: Anthropic rejects an enabled-thinking
# request that also forces a specific tool, and summarize() forces save_summary.
# Effort alone still lowers reasoning spend, which is the whole point.
_NO_EFFORT_PREFIXES = ("claude-sonnet-4-5", "claude-haiku-4-5")
# temperature is removed on Opus 4.7+, Fable 5, and every later model (400 if
# sent). The older models that still accept it — Opus 4.5/4.6, Sonnet 4.5/4.6,
# Haiku 4.5, and earlier — are the deny-list: we keep forwarding temperature
# there for determinism and drop it everywhere else (current + future models).
_KEEPS_TEMPERATURE_PREFIXES = (
    "claude-opus-4-5", "claude-opus-4-6",
    "claude-sonnet-4-5", "claude-sonnet-4-6",
    "claude-haiku-4-5",
)
# Our unified enum has "minimal" (an OpenAI level); Anthropic's lowest is "low".
_EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}

_temperature_warned = False


def _map_effort(reasoning_effort: str) -> str:
    """Map a unified reasoning_effort to Anthropic's ``output_config.effort``,
    raising a named ``ValueError`` on an out-of-enum value (backstop — scribe
    validates first; this beats the bare ``KeyError`` a raw dict lookup raises).
    """
    mapped = _EFFORT_MAP.get(reasoning_effort)
    if mapped is None:
        raise ValueError(
            f"Unknown reasoning_effort {reasoning_effort!r}; "
            f"expected one of {', '.join(_EFFORT_MAP)}."
        )
    return mapped


def _supports_effort(model: str) -> bool:
    # Default safe: everything except the pre-effort deny-list takes effort.
    return not model.startswith(_NO_EFFORT_PREFIXES)


def _drops_temperature(model: str) -> bool:
    # Default safe: drop temperature everywhere except the older models that
    # still accept it, so a current/future temperature-rejecting model can't 400.
    return not model.startswith(_KEEPS_TEMPERATURE_PREFIXES)


def _warn_dropped_temperature(temperature: float, model: str) -> None:
    global _temperature_warned
    if temperature != 1.0 and not _temperature_warned:
        _temperature_warned = True
        console.warn(
            f"{model} does not accept a temperature; ignoring configured "
            f"temperature={temperature} (steer summaries via reasoning effort "
            "and the prompt instead)."
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        self._client = Anthropic()

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> DocumentSummary:
        messages = build_summary_messages(document_text)
        kwargs: dict = dict(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            messages=messages,
            tools=[{
                "name": _SUMMARY_TOOL,
                "description": "Save the document summary.",
                "input_schema": DocumentSummary.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": _SUMMARY_TOOL},
        )
        if reasoning_effort and _supports_effort(model):
            kwargs["output_config"] = {"effort": _map_effort(reasoning_effort)}
        # Current temperature-rejecting models (Opus 4.7+, Fable 5, future) 400 on
        # temperature outright — a property of the model, not of whether we sent
        # effort — so drop it wherever it would 400, independent of
        # reasoning_effort. Only the older deny-listed models still accept it.
        if _drops_temperature(model):
            _warn_dropped_temperature(temperature, model)
            del kwargs["temperature"]
        response = self._client.messages.create(**kwargs)
        return _extract_tool_input(response, _SUMMARY_TOOL, DocumentSummary)

    def classify(
        self,
        prompt: str,
        *,
        model: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        kwargs: dict = dict(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "name": _CLASSIFY_TOOL,
                "description": "Save the classification result.",
                "input_schema": schema.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": _CLASSIFY_TOOL},
        )
        # Temperature-rejecting models 400 on temperature (see summarize) — drop it
        # wherever it would 400. Tag classification reuses the summarizer's model, so
        # a Fable-5/Opus-4.7+ config would otherwise 400 on every classify call.
        if _drops_temperature(model):
            _warn_dropped_temperature(temperature, model)
            del kwargs["temperature"]
        response = self._client.messages.create(**kwargs)
        return _extract_tool_input(response, _CLASSIFY_TOOL, schema)

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        response = self._client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": IMAGE_DESCRIPTION_INSTRUCTIONS},
                ],
            }],
            tools=[{
                "name": _IMAGE_TOOL,
                "description": "Save the image description.",
                "input_schema": VlmDescription.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": _IMAGE_TOOL},
        )
        return _extract_tool_input(response, _IMAGE_TOOL, VlmDescription)


def _extract_tool_input(response, tool_name, model_cls):
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            try:
                return model_cls.model_validate(block.input)
            except ValidationError as e:
                raise RuntimeError(
                    f"Anthropic tool {tool_name!r} input failed schema validation: {e}"
                ) from e
    # A max_tokens truncation cuts the response off before the forced tool call
    # lands, so it surfaces here as a missing tool block. Name it instead of the
    # opaque "no tool use" error — the fix (raise max_tokens) is different.
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"Anthropic response was truncated by max_tokens before the "
            f"{tool_name!r} tool call completed; raise max_tokens and retry."
        )
    raise RuntimeError(
        f"Anthropic response did not include the {tool_name!r} tool call."
    )
