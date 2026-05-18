"""Tests for the .txt fallback chunker (bartleby.ingest.text)."""

from __future__ import annotations

import pytest

from bartleby.ingest.text import chunk_text


def test_chunk_text_short_input_is_single_chunk():
    out = chunk_text("hello world")
    assert out == ["hello world"]


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_overlap_is_honored():
    text = "abcdefghij" * 100  # 1000 chars
    out = chunk_text(text, chunk_size=200, overlap=50)
    assert len(out) >= 2
    # consecutive chunks should share their tail/head bytes
    for prev, curr in zip(out, out[1:]):
        assert prev[-25:] in curr or prev[-50:] in curr


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x" * 200, chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("x" * 200, chunk_size=100, overlap=-1)
