"""Shared prompt builder for the summarizer.

The prompt is provider-agnostic. Schema-enforced JSON makes format
instructions unnecessary — we focus on what makes a good title,
description, and summary.
"""

from __future__ import annotations


_SUMMARY_INSTRUCTIONS = (
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
            "content": f"{_SUMMARY_INSTRUCTIONS}\n\nDOCUMENT:\n{document_text}",
        }
    ]
