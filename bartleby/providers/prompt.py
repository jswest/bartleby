"""Shared prompt builder for the summarizer.

The prompt is provider-agnostic. Schema-enforced JSON makes preamble/format
instructions unnecessary — we focus on what makes a good summary.
"""

from __future__ import annotations


_SUMMARY_INSTRUCTIONS = (
    "Write a concise, self-contained summary of the document below. "
    "Cover its topic, key claims, and structural skeleton. "
    "The summary should be readable on its own, without the original document."
)


def build_summary_messages(document_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{_SUMMARY_INSTRUCTIONS}\n\nDOCUMENT:\n{document_text}",
        }
    ]
