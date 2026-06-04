"""sec2md wrapper — converter for iXBRL EDGAR filings.

Slots in alongside ``docling`` on the HTML branch. Only opted in when
``html_converter = sec2md`` in config (or ``--html-converter sec2md`` on the
CLI), and even then only routed to for files that pass the iXBRL sniff. Other
HTML/HTM files fall through to Docling. See issue #14 for the full scope.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path


SEC2MD_MAX_TOKENS = 400  # headroom under embedder's 512-token cap (matches docling)

_SNIFF_WINDOW_BYTES = 4096
_IXBRL_MARKER = b'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"'


@dataclass
class Sec2mdChunk:
    text: str
    section_heading: str | None
    content_type: str
    page_number: int | None


@dataclass
class Sec2mdResult:
    full_text: str
    page_count: int | None
    chunks: list[Sec2mdChunk]


def _require_sec2md():
    try:
        import sec2md  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "sec2md is not installed in this environment. Install it with one of:\n"
            "  uv tool:    uv tool install --with sec2md --reinstall bartleby\n"
            "  uv venv:    uv pip install 'bartleby[sec2md]'\n"
            "  pip:        pip install 'bartleby[sec2md]'"
        ) from e


def is_ixbrl(path: Path) -> bool:
    """Return True if the first few KB of ``path`` carry the iXBRL namespace marker."""
    try:
        with path.open("rb") as f:
            head = f.read(_SNIFF_WINDOW_BYTES)
    except OSError:
        return False
    return _IXBRL_MARKER in head


def convert(path: Path) -> Sec2mdResult:
    """Convert one iXBRL `.htm`/`.html` filing on disk into chunks.

    Thin wrapper over :func:`convert_bytes` — sec2md doesn't accept file
    paths, so we read the bytes off disk and hand them over.
    """
    return convert_bytes(path.read_bytes())


def convert_bytes(html: bytes) -> Sec2mdResult:
    """Convert iXBRL/SEC HTML bytes into chunks ready for embedding.

    Drives ``parse_filing`` + ``chunk_pages`` with a token cap matched to the
    embedder. Each ``Chunk`` is mapped to a ``Sec2mdChunk`` for the scribe to
    persist via ``insert_document_chunks``. Takes bytes (not a path) so EDGAR
    full-submission inner bodies, which live in memory after unwrapping, can be
    converted without round-tripping through disk.
    """
    _require_sec2md()
    import sec2md
    from bs4 import XMLParsedAsHTMLWarning

    # sec2md's BeautifulSoup invocation triggers a noisy XMLParsedAsHTMLWarning
    # because iXBRL files declare an XML prolog. The warning is purely
    # cosmetic; suppress it for the duration of the parse.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        pages = sec2md.parse_filing(html)
        raw_chunks = sec2md.chunk_pages(pages, chunk_size=SEC2MD_MAX_TOKENS)

    chunks = [
        Sec2mdChunk(
            text=c.embedding_text,
            section_heading=c.header,
            content_type="sec_table" if c.has_table else "sec_text",
            page_number=c.page,
        )
        for c in raw_chunks
        if c.embedding_text and c.embedding_text.strip()
    ]

    page_count = len(pages) or None
    full_text = "\n\f\n".join(p.content for p in pages)

    return Sec2mdResult(
        full_text=full_text,
        page_count=page_count,
        chunks=chunks,
    )
