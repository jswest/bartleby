"""Docling wrapper — one path for `.pdf`, `.html`, `.md`.

Docling handles layout, internal OCR, and structure-aware chunking. The
``HybridChunker`` is initialised with the embedding model's tokenizer and a
``max_tokens`` cap that leaves headroom under the embedder's 512-token limit.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from bartleby.lib.consts import EMBEDDING_MODEL
from bartleby.lib.quiet import is_verbose


DOCLING_MAX_TOKENS = 400  # leaves headroom for heading context (SPEC §5.3)

# Resolution for picture-image extraction. Docling's images_scale is in
# multiples of 72 DPI; 150 DPI matches the pdfplumber backend's PAGE_RENDER_DPI
# so VLM input quality is comparable across converters.
PICTURE_IMAGE_DPI = 150


@dataclass
class DoclingChunk:
    text: str
    section_heading: str | None
    content_type: str | None
    page_number: int | None = None


@dataclass
class DoclingImage:
    png_bytes: bytes
    page_number: int | None
    image_index_on_page: int  # 1-based within a page; matches the image pipeline


@dataclass
class DoclingResult:
    full_text: str
    page_count: int | None
    chunks: list[DoclingChunk]
    images: list[DoclingImage]


def _require_docling():
    try:
        import docling  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "docling is not installed in this environment. Install it with one of:\n"
            "  uv tool:    uv tool install --with docling --reinstall bartleby\n"
            "  uv venv:    uv pip install 'bartleby[docling]'\n"
            "  pip:        pip install 'bartleby[docling]'\n"
            "PDFs can fall back to pdf_converter=pdfplumber in config; "
            "HTML/MD require docling."
        ) from e


@lru_cache(maxsize=2)
def _converter(with_picture_images: bool = False):
    _require_docling()
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Force CPU: Docling's vision models hit float64 ops that MPS rejects on
    # Apple Silicon, failing entire PDFs. CPU is slower but reliable.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU
    )
    # Only populate per-picture raster crops when a caller wants images — it's
    # extra work per page, pointless on text-only ingests.
    if with_picture_images:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = PICTURE_IMAGE_DPI / 72
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def prewarm(with_picture_images: bool = False) -> None:
    """Build the converter + chunker now (idempotent; cached per process).

    The parse pool calls this in each docling worker's initializer so the heavy
    layout/table models load once at startup, not on the worker's first PDF.
    """
    _converter(with_picture_images)
    _chunker()


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


def _first_page(chunk) -> int | None:
    """First page a chunk spans, via Docling's per-item provenance.

    Docling's HybridChunker groups doc items by token budget and can straddle
    page boundaries. We store the chunk's starting page — the schema is one
    integer per chunk and the web view's #page anchor only takes one page
    anyway. Markdown and HTML inputs have no provenance and return None.
    """
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    pages = [
        prov.page_no
        for item in (getattr(meta, "doc_items", None) or [])
        for prov in (getattr(item, "prov", None) or [])
    ]
    return min(pages) if pages else None


def _iter_chunks(chunker, doc) -> Iterable[DoclingChunk]:
    for raw in chunker.chunk(dl_doc=doc):
        contextualized = chunker.contextualize(chunk=raw)
        if not contextualized or not contextualized.strip():
            continue
        yield DoclingChunk(
            text=contextualized,
            section_heading=_section_heading(raw),
            content_type=_content_type(raw),
            page_number=_first_page(raw),
        )


def _iter_picture_images(doc) -> Iterable[tuple[object, int | None, int]]:
    """Yield (pil_image, page_number, image_index_on_page) per extractable picture.

    Pulls each picture's raster via Docling's own ``get_image`` (populated when
    the converter ran with ``generate_picture_images``). Pictures Docling can't
    rasterize — vector figures, some scanned-page art — return ``None`` and are
    skipped; capturing those is a separate, tracked follow-up. The page number
    comes from the picture's first provenance entry, and indices restart at 1
    per page to match the image pipeline's ``image_index_on_page`` convention.
    """
    per_page: dict[int | None, int] = {}
    for pic in doc.pictures:
        image = pic.get_image(doc)
        if image is None:
            continue
        prov = getattr(pic, "prov", None) or []
        page_no = prov[0].page_no if prov else None
        index = per_page.get(page_no, 0) + 1
        per_page[page_no] = index
        yield image, page_no, index


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@contextlib.contextmanager
def _quiet_io():
    """Swallow stdout AND stderr during a Docling ``convert()`` call.

    Docling pulls in RapidOCR (loguru) and transformers/HF (stdlib logging,
    plus rogue ``print()`` calls). Setting log levels doesn't reach loguru
    cleanly, so we capture both streams during conversion. Captured output
    is discarded on both success and exception: the Python exception that
    propagates carries the actionable error (we re-surface it via
    ``console.error``), and docling's internal traceback dumps are
    redundant noise that interleave with our formatted output. Use
    ``--verbose`` (bypasses this entire path) to see everything.

    Rich's progress bars are safe: the Live redraw thread holds the
    original ``sys.stderr`` file object captured at Console construction,
    so it keeps writing past this temporary swap.
    """
    if is_verbose():
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        yield


def convert(path: Path, *, extract_images: bool = False) -> DoclingResult:
    """Convert a `.pdf`, `.html`, or `.md` file with Docling.

    Returns the full document text (Markdown export), the page count (or
    ``None`` for source types without pages), and chunks ready for embedding.
    When ``extract_images`` is set, embedded pictures are rasterized in the same
    pass and returned as ``images`` — no second parse of the source file.
    """
    with _quiet_io():
        result = _converter(extract_images).convert(str(path))
        doc = result.document
        chunks = list(_iter_chunks(_chunker(), doc))
        full_text = doc.export_to_markdown()
        images = []
        if extract_images:
            images = [
                DoclingImage(
                    png_bytes=_png_bytes(image),
                    page_number=page_no,
                    image_index_on_page=index,
                )
                for image, page_no, index in _iter_picture_images(doc)
            ]
    page_count = getattr(doc, "num_pages", None)
    if callable(page_count):
        page_count = page_count()
    if isinstance(page_count, int) and page_count <= 0:
        page_count = None
    return DoclingResult(
        full_text=full_text,
        page_count=page_count,
        chunks=chunks,
        images=images,
    )
