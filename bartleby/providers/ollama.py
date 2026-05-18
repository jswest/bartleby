"""Ollama provider — structured output via the chat API's `format=` schema."""

from __future__ import annotations

import os

import ollama
from pydantic import ValidationError

from bartleby.providers.base import DocumentSummary
from bartleby.providers.prompt import build_summary_messages


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        host = base_url or os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434"
        self._client = ollama.Client(host=host)

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
    ) -> DocumentSummary:
        response = self._client.chat(
            model=model,
            messages=build_summary_messages(document_text),
            format=DocumentSummary.model_json_schema(),
            options={"temperature": temperature},
        )
        content = response.message.content
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        try:
            return DocumentSummary.model_validate_json(content)
        except ValidationError as e:
            raise RuntimeError(
                f"Ollama response failed schema validation: {e}"
            ) from e
