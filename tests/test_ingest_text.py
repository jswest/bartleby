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
    # A non-periodic, whitespace-free body: the digit stream is monotonic, so
    # every 50-char window is unique and appears exactly once. Finding prev's
    # tail inside curr therefore proves a real overlap, not a coincidental
    # repeat. (The old "abcdefghij"*100 input was periodic — prev[-50:]
    # reappeared all over curr even with zero overlap, so the assertion could
    # never fail. No spaces means the chunker's per-piece .strip() can't shift
    # the boundary out from under the slice.)
    text = "".join(f"{i:04d}" for i in range(500))  # 2000 chars, all distinct
    out = chunk_text(text, chunk_size=200, overlap=50)
    assert len(out) >= 2
    # With chunk_size=200/overlap=50, curr starts 150 chars into prev, so prev's
    # last 50 chars are exactly curr's first 50 — present here only because the
    # chunker copied them over.
    for prev, curr in zip(out, out[1:]):
        assert prev[-50:] in curr


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x" * 200, chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("x" * 200, chunk_size=100, overlap=-1)
