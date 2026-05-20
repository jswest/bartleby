"""Provider interface and shared types for LLM + VLM calls."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Schema enforced across providers via structured output.

    A single LLM call returns all three fields so we never pay for the same
    document text three times. Each provider's structured-output mechanism
    (Anthropic tool-use input_schema, OpenAI response_format json_schema,
    Ollama format=) reads ``model_json_schema()`` and enforces every field.
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


class ImageAnalysis(BaseModel):
    """Schema for a single image analyzed by the VLM.

    The ``content_type`` discipline elsewhere in the system needs to know
    which signal is *transcription* (primary source) and which is
    *interpretation*. Splitting ``text`` (OCR) from ``description`` (scene)
    in the model output preserves that distinction all the way to the
    chunks table.
    """

    kind: Literal["text", "scene"] = Field(
        description=(
            "Your judgment of which signal is primary in this image: "
            "'text' if the image is mostly a passage of writing (a page, a sign, "
            "a screenshot of text); 'scene' if it is mostly visual content "
            "(a photo, a diagram, a chart)."
        ),
    )
    text: str = Field(
        description=(
            "Verbatim transcription of every legible piece of text in the image. "
            "Empty string if there is no visible text. Preserve line breaks "
            "where they carry meaning."
        ),
    )
    description: str = Field(
        description=(
            "A factual description of the visual content for an investigative "
            "journalist: subjects, setting, composition, anything a reader "
            "would need to validate a claim against the image. Do not invent "
            "details you cannot see. Empty string if the image is pure text."
        ),
    )
    notes: str = Field(
        description=(
            "What you could not determine and why (illegible regions, ambiguous "
            "subjects, missing context). Empty string if nothing notable."
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

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        media_type: str = "image/jpeg",
    ) -> ImageAnalysis: ...
