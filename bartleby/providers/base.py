"""Provider interface and shared types for LLM + VLM calls."""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

_T = TypeVar("_T", bound=BaseModel)


class DocumentSummary(BaseModel):
    """Schema enforced across providers via structured output.

    A single LLM call returns all four fields so we never pay for the same
    document text multiple times. Each provider's structured-output mechanism
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
    authored_date: str | None = Field(
        default=None,
        description=(
            "The date the document was authored or published, if stated in "
            "the document itself. NOT the date of events described in the "
            "document. NOT an inferred or estimated date. ISO 8601 "
            "YYYY-MM-DD. If only year or only month is known, or if no date "
            "is stated, return null."
        ),
    )


class VlmDescription(BaseModel):
    """Schema the VLM produces for a single image.

    Description-only. OCR (the ``text`` field on the orchestrator-level
    ``ImageAnalysis``) is owned by Tesseract — the VLM is never asked to
    transcribe. Splitting the responsibilities keeps the VLM output bounded
    (~200 words) so per-image latency stays predictable on local models.
    """

    description: str = Field(
        description=(
            "A factual description of the visual content for an investigative "
            "journalist: subjects, setting, composition, chart type, layout. "
            "Mention salient text (titles, axis labels) only where needed for "
            "coherence; never produce a full transcription."
        ),
    )
    notes: str = Field(
        description=(
            "What you could not determine and why (illegible regions, ambiguous "
            "subjects, missing context). Empty string if nothing notable."
        ),
    )


class ImageAnalysis(BaseModel):
    """Orchestrator-level merged result for one image.

    The image pipeline runs Tesseract first; if Tesseract returns substantial
    text at decent confidence the image is classified ``kind='text'`` and the
    VLM is never called (``text`` is the Tesseract output; ``description`` and
    ``notes`` stay empty). Otherwise the image is ``kind='scene'``, the VLM
    fills ``description`` + ``notes``, and ``text`` stays empty.

    This shape is what gets persisted to ``images.analysis_json`` and what the
    chunker reads to decide which ``image_*`` content_type chunks to emit.
    """

    kind: Literal["text", "scene"] = Field(
        description="Set by the orchestrator based on Tesseract dispositioning.",
    )
    text: str = Field(
        description="Tesseract OCR output (empty for scene-images).",
    )
    description: str = Field(
        description="VLM scene description (empty for text-images).",
    )
    notes: str = Field(
        description="VLM caveats (empty for text-images).",
    )


class Provider(Protocol):
    name: str

    def summarize(
        self,
        document_text: str,
        *,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> DocumentSummary: ...

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        model: str,
        media_type: str = "image/jpeg",
    ) -> VlmDescription: ...

    def classify(
        self,
        prompt: str,
        *,
        model: str,
        schema: type[_T],
        temperature: float = 0.0,
    ) -> _T:
        """Structured-output call for arbitrary Pydantic schemas.

        Used wherever the codebase needs typed JSON out of an LLM that
        isn't `summarize`/`analyze_image` — e.g. tag classification.
        """
        ...
