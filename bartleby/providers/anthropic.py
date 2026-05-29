"""Anthropic provider — structured output via tool-use."""

from __future__ import annotations

import base64

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    build_summary_messages,
)


_SUMMARY_TOOL = "save_summary"
_IMAGE_TOOL = "save_image_description"
_CLASSIFY_TOOL = "save_classification"


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
    ) -> DocumentSummary:
        messages = build_summary_messages(document_text)
        response = self._client.messages.create(
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
        return _extract_tool_input(response, _SUMMARY_TOOL, DocumentSummary)

    def classify(
        self,
        prompt: str,
        *,
        model: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        response = self._client.messages.create(
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
