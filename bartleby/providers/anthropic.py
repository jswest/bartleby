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

# Reasoning effort is GA via output_config.effort on Opus 4.5+ and Sonnet 4.6;
# it 400s on Sonnet 4.5 / Haiku 4.5 and earlier, so we only send it for the
# models below and leave older models on their default behavior. We deliberately
# do NOT pair it with a thinking block: Anthropic rejects an enabled-thinking
# request that also forces a specific tool, and summarize() forces save_summary.
# Effort alone still lowers reasoning spend, which is the whole point.
_EFFORT_MODEL_PREFIXES = (
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6",
)
# temperature is removed on Opus 4.7+ (400 if sent); on the older effort-capable
# models it's still accepted, so we keep forwarding it there for determinism.
_NO_TEMPERATURE_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8")
# Our unified enum has "minimal" (an OpenAI level); Anthropic's lowest is "low".
_EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}

_temperature_warned = False


def _supports_effort(model: str) -> bool:
    return model.startswith(_EFFORT_MODEL_PREFIXES)


def _drops_temperature(model: str) -> bool:
    return model.startswith(_NO_TEMPERATURE_PREFIXES)


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
            kwargs["output_config"] = {"effort": _EFFORT_MAP[reasoning_effort]}
        # Opus 4.7+ rejects temperature outright — a property of the model, not of
        # whether we sent effort — so drop it wherever it would 400, independent of
        # reasoning_effort. The older effort-capable models still accept it.
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
        # Opus 4.7+ rejects temperature outright (see summarize) — drop it wherever
        # it would 400. Tag classification reuses the summarizer's model, so a
        # 4.7+ config would otherwise 400 on every classify call.
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
    raise RuntimeError(
        f"Anthropic response did not include the {tool_name!r} tool call."
    )
