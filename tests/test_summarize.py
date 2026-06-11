"""Tests for the summarizer (Pydantic enforcement, truncation note)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bartleby.ingest.summarize import (
    count_tokens, normalize_authored_date, summarize,
)
from bartleby.providers.base import DocumentSummary


@dataclass
class FakeProvider:
    """Captures inputs so tests can inspect what the LLM would have seen."""
    captured_text: str | None = None
    captured_model: str | None = None
    captured_temperature: float | None = None
    captured_reasoning_effort: str | None = None
    return_title: str = "Doc Title"
    return_description: str = "A one-line hook describing the document."
    return_text: str = "this is a summary"
    return_authored_date: str | None = None

    name = "fake"

    def summarize(self, document_text: str, *, model: str, temperature: float,
                  reasoning_effort: str | None = None):
        self.captured_text = document_text
        self.captured_model = model
        self.captured_temperature = temperature
        self.captured_reasoning_effort = reasoning_effort
        return DocumentSummary(
            title=self.return_title,
            description=self.return_description,
            text=self.return_text,
            authored_date=self.return_authored_date,
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
    assert p.captured_text == "short document"


def test_summarize_forwards_reasoning_effort_to_provider():
    p = FakeProvider()
    summarize(
        "short document",
        provider=p, model="m", temperature=0.0,
        max_summarize_tokens=1000, reasoning_effort="low",
    )
    assert p.captured_reasoning_effort == "low"


def test_summarize_reasoning_effort_defaults_to_none():
    p = FakeProvider()
    summarize(
        "short document",
        provider=p, model="m", temperature=0.0, max_summarize_tokens=1000,
    )
    assert p.captured_reasoning_effort is None


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


def test_summarize_passes_through_valid_authored_date():
    p = FakeProvider(return_authored_date="2024-03-15")
    result = summarize(
        "doc", provider=p, model="m", temperature=0.0,
        max_summarize_tokens=100,
    )
    assert result.authored_date == "2024-03-15"


@pytest.mark.parametrize("raw", [
    None, "", "Q3 2024", "2024", "2024-03", "March 15, 2024",
    "2024/03/15", "  ", "2024-13-01", "2024-02-30",
])
def test_summarize_drops_malformed_authored_date(raw):
    assert normalize_authored_date(raw) is None


def test_summarize_accepts_iso_date():
    assert normalize_authored_date("2024-03-15") == "2024-03-15"
    assert normalize_authored_date(" 2024-03-15 ") == "2024-03-15"
