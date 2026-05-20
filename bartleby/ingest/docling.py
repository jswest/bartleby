"""Docling wrapper — one path for `.pdf`, `.html`, `.md`.

Docling handles layout, internal OCR, and structure-aware chunking. The
``HybridChunker`` is initialised with the embedding model's tokenizer and a
``max_tokens`` cap that leaves headroom under the embedder's 512-token limit.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from bartleby.lib.consts import EMBEDDING_MODEL
from bartleby.lib.quiet import is_verbose


DOCLING_MAX_TOKENS = 400  # leaves headroom for heading context (SPEC §5.3)


@dataclass
class DoclingChunk:
    text: str
    section_heading: str | None
    content_type: str | None


@dataclass
class DoclingResult:
    full_text: str
    page_count: int | None
    chunks: list[DoclingChunk]


def _require_docling():
    try:
        import docling  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The 'docling' extra is required for this code path "
            "(install with `uv pip install 'bartleby[docling]'` or "
            "`pip install 'bartleby[docling]'`). Either install it or switch "
            "to the pdfplumber backend for PDFs."
        ) from e


@lru_cache(maxsize=1)
def _converter():
    _require_docling()
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


@lru_cache(maxsize=1)
def _chunker():
    _require_docling()
    from docling.chunking import HybridChunker

    return HybridChunker(tokenizer=EMBEDDING_MODEL, max_tokens=DOCLING_MAX_TOKENS)


def _section_heading(chunk) -> str | None:
    """Pull the deepest section heading from a Docling chunk's metadata."""
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    headings = getattr(meta, "headings", None)
    if headings:
        return headings[-1]
    return None


def _content_type(chunk) -> str | None:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    doc_items = getattr(meta, "doc_items", None) or []
    if not doc_items:
        return None
    label = getattr(doc_items[0], "label", None)
    return str(label) if label is not None else None


def _iter_chunks(chunker, doc) -> Iterable[DoclingChunk]:
    for raw in chunker.chunk(dl_doc=doc):
        contextualized = chunker.contextualize(chunk=raw)
        if not contextualized or not contextualized.strip():
            continue
        yield DoclingChunk(
            text=contextualized,
            section_heading=_section_heading(raw),
            content_type=_content_type(raw),
        )


@contextlib.contextmanager
def _quiet_io():
    """Swallow stdout AND stderr during a Docling ``convert()`` call.

    Docling pulls in RapidOCR (loguru) and transformers/HF (stdlib logging,
    plus rogue ``print()`` calls). Setting log levels doesn't reach loguru
    cleanly, so we capture both streams during conversion. The captured
    output is dumped to the real stderr only if the wrapped call raises,
    so diagnostics survive failures.

    Rich's progress bars are safe: the Live redraw thread holds the
    original ``sys.stderr`` file object captured at Console construction,
    so it keeps writing past this temporary swap.
    """
    if is_verbose():
        yield
        return
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            yield
    except Exception:
        sys.stderr.write(out_buf.getvalue())
        sys.stderr.write(err_buf.getvalue())
        raise


def convert(path: Path) -> DoclingResult:
    """Convert a `.pdf`, `.html`, or `.md` file with Docling.

    Returns the full document text (Markdown export), the page count (or
    ``None`` for source types without pages), and chunks ready for embedding.
    """
    with _quiet_io():
        result = _converter().convert(str(path))
        doc = result.document
        chunks = list(_iter_chunks(_chunker(), doc))
        full_text = doc.export_to_markdown()
    page_count = getattr(doc, "num_pages", None)
    if callable(page_count):
        page_count = page_count()
    if isinstance(page_count, int) and page_count <= 0:
        page_count = None
    return DoclingResult(
        full_text=full_text,
        page_count=page_count,
        chunks=chunks,
    )
