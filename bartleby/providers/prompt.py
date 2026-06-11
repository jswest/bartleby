"""Shared prompt builder for the summarizer.

The prompt is provider-agnostic. Schema-enforced JSON makes format
instructions unnecessary — we focus on what makes a good title,
description, and summary.
"""

from __future__ import annotations

from bartleby.providers.base import DocumentSummary


# Single source of truth: the per-field rules live on DocumentSummary's Field
# descriptions, which already enforce structured output for every provider.
# Render them into the prompt rather than hand-copying — Ollama's format=
# grammar channel never shows the model the schema descriptions as text, so the
# prompt must keep carrying them.
SUMMARY_INSTRUCTIONS = (
    "Read the document below and produce four things in one pass:\n"
    + "\n".join(
        f"  - {name}: {field.description}"
        for name, field in DocumentSummary.model_fields.items()
    )
)


def build_summary_messages(document_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{SUMMARY_INSTRUCTIONS}\n\nDOCUMENT:\n{document_text}",
        }
    ]


IMAGE_DESCRIPTION_INSTRUCTIONS = (
    "You are describing an image so an investigative journalist can decide "
    "whether to look at the original. OCR is handled separately by Tesseract — "
    "do NOT transcribe text. Your job is interpretation, not transcription.\n\n"
    "Return two things:\n"
    "  - description: a factual, observational paragraph describing the visual "
    "content. Cover subjects, setting, composition, layout, chart type (if "
    "any), and what's notable. You may mention salient text — titles, axis "
    "labels, captions, on-screen labels — where it's needed to make the "
    "description coherent, but quote sparingly and never produce a full "
    "transcription. Keep it under ~200 words.\n"
    "  - notes: anything you could not determine and why (illegible regions, "
    "ambiguous subjects, missing context). Empty string if nothing notable."
)
