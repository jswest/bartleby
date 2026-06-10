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

# A filing must resolve at least this many internal TOC anchors to in-document
# targets before it is split into sections (#254). Splitting into one section is
# just the whole file with extra ceremony; the win starts when a monolith
# fractures into many independently-summarizable units.
_MIN_SECTIONS_TO_SPLIT = 2


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


@dataclass
class Sec2mdSection:
    """One anchor-delimited slice of a filing (#254).

    ``anchor_id`` is the HTML ``id`` the table of contents linked to; ``title``
    is the TOC link text (free semantic labelling); ``order`` is its position in
    the TOC. ``result`` is the independently-converted body of just that slice —
    its own chunks, its own (eventual) summary.
    """
    anchor_id: str
    title: str | None
    order: int
    result: Sec2mdResult


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


def convert_sections(path: Path) -> list[Sec2mdSection]:
    """Split a TOC-anchored filing into per-section conversions (#254).

    Returns one :class:`Sec2mdSection` per internal anchor the table of contents
    resolves to an in-document target, in TOC order, each with its slice
    converted independently via :func:`convert_bytes`. Returns ``[]`` when the
    filing carries no usable TOC (fewer than ``_MIN_SECTIONS_TO_SPLIT`` resolved
    anchors) — the caller then ingests it whole, exactly as before.

    The split is purely structural: the raw HTML body is sliced at the
    top-level ancestors of each anchor target, so every byte of content lands in
    exactly one section and nothing is duplicated or dropped.
    """
    return _convert_sections_bytes(path.read_bytes())


def _convert_sections_bytes(html: bytes) -> list[Sec2mdSection]:
    _require_sec2md()
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "html.parser")

    body = soup.body or soup
    targets = _resolve_toc_targets(soup, body)
    if len(targets) < _MIN_SECTIONS_TO_SPLIT:
        return []

    sections: list[Sec2mdSection] = []
    for order, (anchor_id, title, target_el) in enumerate(targets):
        start = _top_level_ancestor(target_el, body)
        next_start = (
            _top_level_ancestor(targets[order + 1][2], body)
            if order + 1 < len(targets) else None
        )
        fragment = _slice_between(start, next_start)
        if not fragment.strip():
            continue
        section_html = f"<html><body>{fragment}</body></html>".encode("utf-8")
        result = convert_bytes(section_html)
        if not result.chunks:
            # An anchor pointing at an empty/decorative slice carries nothing to
            # index — skip it rather than persist a content-free section row.
            continue
        sections.append(Sec2mdSection(
            anchor_id=anchor_id, title=title or None, order=order, result=result,
        ))

    # If the slices collapsed to a single (or no) real section, splitting buys
    # nothing — let the caller ingest the file whole.
    if len(sections) < _MIN_SECTIONS_TO_SPLIT:
        return []
    return sections


def _resolve_toc_targets(soup, body) -> list[tuple[str, str, object]]:
    """The TOC's internal anchors that resolve to in-document targets, in order.

    Each entry is ``(anchor_id, link_text, target_element)``. Only the first
    link to a given id is kept (a TOC that lists a section twice still yields one
    section), and only ids that actually exist as a target within the body
    count — a dangling ``#ref`` is no section boundary.
    """
    seen: set[str] = set()
    targets: list[tuple[str, str, object]] = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if not href.startswith("#") or len(href) < 2:
            continue
        anchor_id = href[1:]
        if anchor_id in seen:
            continue
        target = body.find(id=anchor_id)
        if target is None:
            continue
        seen.add(anchor_id)
        targets.append((anchor_id, a.get_text(strip=True), target))
    return targets


def _top_level_ancestor(el, body):
    """The body-level block that contains ``el`` (the slice boundary unit)."""
    while el.parent is not None and el.parent is not body:
        el = el.parent
    return el


def _slice_between(start, end) -> str:
    """Serialize every top-level sibling from ``start`` up to (not incl) ``end``."""
    parts: list[str] = []
    node = start
    while node is not None and node is not end:
        if getattr(node, "name", None) is not None:
            parts.append(str(node))
        node = node.find_next_sibling()
    return "".join(parts)
