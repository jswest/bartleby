"""OpenAI provider — structured output via the Pydantic `parse` helper."""

from __future__ import annotations

import base64

from openai import OpenAI
from pydantic import BaseModel

from bartleby.lib import console
from bartleby.providers.base import DocumentSummary, VlmDescription
from bartleby.providers.prompt import (
    IMAGE_DESCRIPTION_INSTRUCTIONS,
    build_summary_messages,
)

# OpenAI's models (the GPT-5 family) accept only the default temperature (1.0);
# any other value is rejected outright. We never send the parameter — the API
# default stands — and warn once per process if the caller configured a value
# we're dropping, so a non-default temperature in config doesn't read as honored.
_temperature_warned = False


def _drop_temperature(temperature: float) -> None:
    global _temperature_warned
    if temperature != 1.0 and not _temperature_warned:
        _temperature_warned = True
        console.warn(
            "OpenAI models accept only the default temperature (1.0); "
            f"ignoring configured temperature={temperature}."
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
        reasoning_effort: str | None = None,
    ) -> DocumentSummary:
        _drop_temperature(temperature)
        # gpt-5 accepts minimal | low | medium | high directly. Pass it only when
        # configured so non-reasoning models aren't handed an unknown parameter.
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        response = self._client.chat.completions.parse(
            model=model,
            messages=build_summary_messages(document_text),
            response_format=DocumentSummary,
            **extra,
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
        _drop_temperature(temperature)
        response = self._client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        return _require_parsed(response, schema)

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        temperature: float,
        media_type: str = "image/jpeg",
    ) -> VlmDescription:
        _drop_temperature(temperature)
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
