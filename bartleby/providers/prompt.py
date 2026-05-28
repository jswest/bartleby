"""Shared prompt builder for the summarizer.

The prompt is provider-agnostic. Schema-enforced JSON makes format
instructions unnecessary — we focus on what makes a good title,
description, and summary.
"""

from __future__ import annotations


SUMMARY_INSTRUCTIONS = (
    "Read the document below and produce three things in one pass:\n"
    "  - title: a short human-readable title (no filename, no quotes, "
    "no trailing punctuation, 60 characters or fewer).\n"
    "  - description: a one-sentence hook (~20 words) that tells a reader "
    "what this document is and why they might care.\n"
    "  - text: a concise, self-contained summary covering the topic, key "
    "claims, and structural skeleton. Readable on its own, without the "
    "original document."
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
