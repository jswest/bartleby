"""Tests for the pure-Python markdown chunker (bartleby.ingest.chunk)."""

from __future__ import annotations

from bartleby.ingest.chunk import chunk_markdown_string


def test_short_unheaded_markdown_is_one_chunk():
    rows = chunk_markdown_string("Just a short paragraph with no header.")
    assert len(rows) == 1
    assert rows[0].section_heading is None
    assert "short paragraph" in rows[0].text


def test_atx_headers_become_section_headings():
    md = "# Intro\nIntro body.\n\n## Methods\nMethod body."
    rows = chunk_markdown_string(md)
    headings = [r.section_heading for r in rows]
    assert "Intro" in headings
    assert "Methods" in headings


def test_empty_markdown_yields_no_rows():
    assert chunk_markdown_string("") == []
    assert chunk_markdown_string("   \n\n   ") == []


def test_oversize_paragraph_is_hard_split():
    md = "x" * 5000
    rows = chunk_markdown_string(md)
    # Each row's text should fit under a reasonable cap (1600 default + slack).
    assert all(len(r.text) <= 1700 for r in rows)
    assert len(rows) >= 2


def test_paragraphs_pack_into_one_chunk_when_they_fit():
    md = "p1.\n\np2.\n\np3."
    rows = chunk_markdown_string(md)
    assert len(rows) == 1
    assert "p1" in rows[0].text and "p3" in rows[0].text
