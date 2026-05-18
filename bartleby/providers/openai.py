"""OpenAI provider — structured output via the Pydantic `parse` helper."""

from __future__ import annotations

from openai import OpenAI

from bartleby.providers.base import DocumentSummary
from bartleby.providers.prompt import build_summary_messages


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        self._client = OpenAI()

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
    ) -> DocumentSummary:
        response = self._client.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=build_summary_messages(document_text),
            response_format=DocumentSummary,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            refusal = response.choices[0].message.refusal
            raise RuntimeError(
                f"OpenAI returned no parsed payload "
                f"(refusal={refusal!r})."
            )
        return parsed
