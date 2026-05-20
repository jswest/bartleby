"""Tests for the summarizer (Pydantic enforcement, truncation note)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bartleby.ingest.summarize import count_tokens, summarize
from bartleby.providers.base import DocumentSummary


@dataclass
class FakeProvider:
    """Captures inputs so tests can inspect what the LLM would have seen."""
    captured_text: str | None = None
    captured_model: str | None = None
    captured_temperature: float | None = None
    return_title: str = "Doc Title"
    return_description: str = "A one-line hook describing the document."
    return_text: str = "this is a summary"

    name = "fake"

    def summarize(self, document_text: str, *, model: str, temperature: float):
        self.captured_text = document_text
        self.captured_model = model
        self.captured_temperature = temperature
        return DocumentSummary(
            title=self.return_title,
            description=self.return_description,
            text=self.return_text,
        )


def test_summarize_short_doc_no_truncation_note():
    p = FakeProvider()
    result = summarize(
        "short document",
        provider=p, model="m", temperature=0.0,
        max_summarize_tokens=1000,
    )
    assert result.title == "Doc Title"
    assert result.description == "A one-line hook describing the document."
    assert result.text == "this is a summary"
    assert result.truncated_from_tokens is None
    assert p.captured_text == "short document"


def test_summarize_long_doc_truncates_input_and_appends_note():
    # Produce a doc that exceeds 10 tokens.
    long_text = "word " * 5000  # roughly 5000 tokens
    total = count_tokens(long_text)
    assert total > 10  # sanity

    p = FakeProvider(return_text="summary body")
    result = summarize(
        long_text,
        provider=p, model="m", temperature=0.0,
        max_summarize_tokens=10,
    )

    assert result.truncated_from_tokens == total
    # Truncation note is appended to text only (title/description are unaffected).
    assert result.title == "Doc Title"
    assert result.text.startswith("summary body")
    assert "first 10 tokens" in result.text
    assert f"{total:,}-token document" in result.text
    # Verify the provider actually received the truncated input.
    assert count_tokens(p.captured_text) <= 10


def test_summarize_rejects_empty_document():
    with pytest.raises(ValueError):
        summarize(
            "",
            provider=FakeProvider(), model="m", temperature=0.0,
            max_summarize_tokens=100,
        )
