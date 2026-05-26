"""Ollama provider — structured output via the chat API's `format=` schema."""

from __future__ import annotations

import os

import ollama
from pydantic import BaseModel, ValidationError

from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    build_summary_messages,
)


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
        return _validate(response.message.content, DocumentSummary)

    def classify(
        self,
        prompt: str,
        *,
        model: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
            options={"temperature": temperature},
        )
        return _validate(response.message.content, schema)

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        # Ollama's chat API takes raw image bytes (or paths) via `images=`.
        response = self._client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": IMAGE_DESCRIPTION_INSTRUCTIONS,
                "images": [image_bytes],
            }],
            format=VlmDescription.model_json_schema(),
        )
        return _validate(response.message.content, VlmDescription)


def _validate(content, model_cls):
    if not content:
        raise RuntimeError(f"Ollama returned an empty response for {model_cls.__name__}.")
    try:
        return model_cls.model_validate_json(content)
    except ValidationError as e:
        raise RuntimeError(
            f"Ollama response failed {model_cls.__name__} schema validation: {e}"
        ) from e
