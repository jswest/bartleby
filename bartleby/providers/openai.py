"""OpenAI provider — structured output via the Pydantic `parse` helper."""

from __future__ import annotations

import base64

from openai import OpenAI
from pydantic import BaseModel

from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    build_summary_messages,
)


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
        return _require_parsed(response, DocumentSummary)

    def classify(
        self,
        prompt: str,
        *,
        model: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        response = self._client.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        return _require_parsed(response, schema)

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        response = self._client.chat.completions.parse(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_DESCRIPTION_INSTRUCTIONS},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                ],
            }],
            response_format=VlmDescription,
        )
        return _require_parsed(response, VlmDescription)


def _require_parsed(response, model_cls):
    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise RuntimeError(
            f"OpenAI returned no parsed payload for {model_cls.__name__} "
            f"(refusal={refusal!r})."
        )
    return parsed
