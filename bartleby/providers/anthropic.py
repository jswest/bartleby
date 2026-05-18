"""Anthropic provider — structured output via tool-use."""

from __future__ import annotations

from anthropic import Anthropic
from pydantic import ValidationError

from bartleby.providers.base import DocumentSummary
from bartleby.providers.prompt import build_summary_messages


_TOOL_NAME = "save_summary"


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
                "name": _TOOL_NAME,
                "description": "Save the document summary.",
                "input_schema": DocumentSummary.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                try:
                    return DocumentSummary.model_validate(block.input)
                except ValidationError as e:
                    raise RuntimeError(
                        f"Anthropic returned tool input that failed schema validation: {e}"
                    ) from e

        raise RuntimeError(
            f"Anthropic response did not include the {_TOOL_NAME!r} tool call."
        )
