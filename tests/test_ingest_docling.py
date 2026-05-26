"""Unit tests for the Docling adapter's page-number extraction.

These tests don't load Docling itself — they exercise the meta-shape parser
against namespace objects shaped like Docling's chunk metadata. Real
Docling integration is covered by `bartleby scribe` runs against PDFs.
"""

from __future__ import annotations

from types import SimpleNamespace

from bartleby.ingest.docling import _first_page


def _chunk(*page_lists: list[int]) -> SimpleNamespace:
    """Build a fake Docling chunk where each doc_item has its own prov list."""
    doc_items = [
        SimpleNamespace(prov=[SimpleNamespace(page_no=p) for p in pages])
        for pages in page_lists
    ]
    return SimpleNamespace(meta=SimpleNamespace(doc_items=doc_items))


def test_first_page_returns_min_across_all_provenance():
    """A chunk that straddles pages 3, 4, 5 starts on page 3."""
    chunk = _chunk([3, 4], [4, 5])
    assert _first_page(chunk) == 3


def test_first_page_single_page_chunk():
    chunk = _chunk([2])
    assert _first_page(chunk) == 2


def test_first_page_none_when_no_provenance():
    """Markdown and HTML inputs produce chunks without prov data."""
    chunk = _chunk()
    assert _first_page(chunk) is None


def test_first_page_none_when_doc_items_have_empty_prov():
    chunk = SimpleNamespace(
        meta=SimpleNamespace(doc_items=[SimpleNamespace(prov=[])])
    )
    assert _first_page(chunk) is None


def test_first_page_none_when_meta_missing():
    chunk = SimpleNamespace()
    assert _first_page(chunk) is None
