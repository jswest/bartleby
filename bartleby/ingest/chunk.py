"""Convert + chunk facade for non-PDF, non-image files.

PDFs are dispatched by ``bartleby.commands.scribe`` directly to the pdfplumber
or docling backends; standalone image files go through the image pipeline.
This module covers HTML / Markdown (via docling) and plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import filetype

from bartleby.ingest import docling as docling_pipeline
from bartleby.ingest import text as text_pipeline


PDF_EXTENSIONS = {".pdf"}
DOCLING_ONLY_EXTENSIONS = {".html", ".htm", ".md"}     # docling is the only converter
TEXT_EXTENSIONS = {".txt"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | DOCLING_ONLY_EXTENSIONS | TEXT_EXTENSIONS | IMAGE_EXTENSIONS
)


def resolve_extension(path: Path) -> str | None:
    """Return the supported extension to treat ``path`` as, or ``None``.

    The filename extension wins whenever it is one we support: a recognized
    extension is trusted as-is and we never second-guess it from content (so a
    ``.txt`` that happens to start with PDF bytes stays text). Sniffing kicks in
    *only* when the extension gives us nothing usable — missing, or not one we
    recognize — in which case we read the file's magic bytes via ``filetype``.

    Returns a leading-dot, lowercased extension drawn from
    ``SUPPORTED_EXTENSIONS``, or ``None`` when the file is an unsupported type or
    its type cannot be determined at all.
    """
    ext = path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    kind = filetype.guess(str(path))
    if kind is None:
        return None
    sniffed = f".{kind.extension.lower()}"
    return sniffed if sniffed in SUPPORTED_EXTENSIONS else None


# Soft cap for agent-authored markdown chunks. Picked to stay well under the
# embedder's 512-token max sequence; characters are a rough proxy.
_MD_CHUNK_CHARS = 1600


@dataclass
class ChunkRow:
    text: str
    section_heading: str | None
    content_type: str | None
    page_number: int | None = None


@dataclass
class ConversionResult:
    full_text: str
    page_count: int | None
    chunks: list[ChunkRow]


def convert_and_chunk(path: Path) -> ConversionResult:
    ext = path.suffix.lower()
    if ext in DOCLING_ONLY_EXTENSIONS:
        result = docling_pipeline.convert(path)
        rows = [
            ChunkRow(text=c.text, section_heading=c.section_heading,
                     content_type=c.content_type, page_number=c.page_number)
            for c in result.chunks
        ]
        return ConversionResult(
            full_text=result.full_text,
            page_count=result.page_count,
            chunks=rows,
        )
    if ext in TEXT_EXTENSIONS:
        full = text_pipeline.read_text(path)
        rows = [
            ChunkRow(text=t, section_heading=None, content_type=None)
            for t in text_pipeline.chunk_text(full)
        ]
        return ConversionResult(full_text=full, page_count=None, chunks=rows)
    raise ValueError(f"Unsupported extension: {ext}")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown_string(markdown: str) -> list[ChunkRow]:
    """Chunk agent-authored markdown by ATX headers, then by size within sections.

    Each chunk carries the deepest enclosing heading. No Docling involvement —
    we trust the agent's markdown is structurally simple and don't need
    layout-aware parsing for it.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(2), []))
        else:
            sections[-1][1].append(line)

    rows: list[ChunkRow] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for piece in _split_by_size(body, _MD_CHUNK_CHARS):
            text = piece.strip()
            if text:
                rows.append(ChunkRow(text=text, section_heading=heading,
                                     content_type=None))
    return rows


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries first; only hard-split if a paragraph alone
    exceeds the size cap."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        if len(para) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(para), max_chars):
                out.append(para[i:i + max_chars])
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > max_chars:
            out.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out
