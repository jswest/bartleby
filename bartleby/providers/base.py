"""Provider interface and shared types for LLM summarization."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Pydantic schema enforced across all three providers (SPEC §5.3.1).

    A single LLM call returns all three fields so we never pay for the same
    document text three times. The provider's structured-output mechanism
    (Anthropic tool-use input_schema, OpenAI response_format json_schema,
    Ollama format=) reads ``model_json_schema()`` and enforces all fields.
    """

    title: str = Field(
        description=(
            "A short human-readable title for the document. Plain text only — "
            "no quotes, no trailing punctuation, no filename. 60 characters or fewer."
        ),
    )
    description: str = Field(
        description=(
            "A one-sentence hook that tells a reader what the document is and "
            "why they might care. Aim for ~20 words; never exceed 200 characters."
        ),
    )
    text: str = Field(
        description=(
            "A concise, self-contained summary of the document covering its "
            "topic, key claims, and structural skeleton. Readable on its own."
        ),
    )


class Provider(Protocol):
    name: str

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
    ) -> DocumentSummary: ...
