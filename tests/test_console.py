"""Unit tests for the console helpers used by the scribe progress bar."""

from __future__ import annotations

from bartleby.lib.console import truncate_filename


def test_truncate_filename_passes_short_names_through():
    assert truncate_filename("report.pdf") == "report.pdf"


def test_truncate_filename_respects_exact_max():
    name = "x" * 40
    assert truncate_filename(name, max_len=40) == name


def test_truncate_filename_middle_truncates_long_names():
    name = "A_very_long_scraper_mangled_filename_that_keeps_going.PDF"
    out = truncate_filename(name, max_len=40)
    assert len(out) == 40
    assert "…" in out
    # Head and tail (extension) both survive the middle ellipsis.
    assert out.startswith("A_very_long")
    assert out.endswith(".PDF")
