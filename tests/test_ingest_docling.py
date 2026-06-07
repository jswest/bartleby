"""Unit tests for the Docling adapter's page-number and picture extraction.

These tests don't load Docling itself — they exercise the meta-shape parsers
against namespace objects shaped like Docling's chunk metadata and picture
items. Real Docling integration is covered by `bartleby scribe` runs against
PDFs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bartleby.ingest.docling import (
    _first_page,
    _iter_picture_images,
    _normalize_device,
)
from bartleby.lib.consts import DEFAULT_DOCLING_DEVICE


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


def _pic(page_no: int | None, image: object) -> SimpleNamespace:
    """A fake PictureItem: prov carries the page, get_image returns the raster."""
    prov = [SimpleNamespace(page_no=page_no)] if page_no is not None else []
    return SimpleNamespace(prov=prov, get_image=lambda doc: image)


def _doc(*pics: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(pictures=list(pics))


def test_iter_picture_images_indexes_restart_per_page():
    """Two pictures on page 1, one on page 2 → indices 1, 2, 1."""
    a, b, c = object(), object(), object()
    doc = _doc(_pic(1, a), _pic(1, b), _pic(2, c))
    assert list(_iter_picture_images(doc)) == [
        (a, 1, 1), (b, 1, 2), (c, 2, 1),
    ]


def test_iter_picture_images_skips_unrasterizable():
    """A picture whose get_image returns None is dropped and burns no index."""
    a, c = object(), object()
    doc = _doc(_pic(1, a), _pic(1, None), _pic(1, c))
    assert list(_iter_picture_images(doc)) == [(a, 1, 1), (c, 1, 2)]


def test_iter_picture_images_page_none_when_no_provenance():
    """A picture without provenance still rasterizes, with page_number None."""
    a = object()
    doc = _doc(_pic(None, a))
    assert list(_iter_picture_images(doc)) == [(a, None, 1)]


# --- docling_device knob -----------------------------------------------------

def test_normalize_device_empty_falls_back_to_default():
    """Absent/empty config means "use the default", not an error."""
    assert _normalize_device(None) == DEFAULT_DOCLING_DEVICE
    assert _normalize_device("") == DEFAULT_DOCLING_DEVICE


def test_normalize_device_accepts_cpu_and_cuda():
    assert _normalize_device("cpu") == "cpu"
    assert _normalize_device("cuda") == "cuda"


def test_normalize_device_is_case_and_whitespace_insensitive():
    assert _normalize_device("  CUDA ") == "cuda"


def test_normalize_device_rejects_unknown():
    """A typo'd device fails loudly rather than silently running on CPU."""
    with pytest.raises(ValueError, match="auto"):
        _normalize_device("auto")
