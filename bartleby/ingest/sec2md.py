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
from typing import NamedTuple


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


# A synthetic anchor id for the front-matter section that precedes the first TOC
# target (#254 rework). EDGAR cover pages carry the registrant name, CIK, period,
# ticker and shares-outstanding — real content that must land in a section rather
# than be dropped. The leading/trailing underscores keep it from colliding with a
# genuine HTML id a filing might use as a TOC target.
_PREAMBLE_ANCHOR_ID = "__preamble__"
_PREAMBLE_TITLE = "Preamble"


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
    targets, toc_link_els = _resolve_toc_targets(soup, body)
    if len(targets) < _MIN_SECTIONS_TO_SPLIT:
        return []

    # Slice boundaries are the top-level ancestors of each target, IN DOCUMENT
    # ORDER (targets are already so ordered). Each slice ends at the next
    # target's boundary; only the genuine last target runs to end-of-body. A
    # synthetic "preamble" section captures any body content before the first
    # target (an EDGAR cover page: registrant name, CIK, period, ticker), which
    # would otherwise land in no section and vanish from FTS/embeddings.
    boundaries = [
        (anchor_id, title, _top_level_ancestor(target_el, body))
        for anchor_id, title, target_el in targets
    ]
    first_start = boundaries[0][2]
    # The TOC nav block(s) sit before the first target but are navigation, not
    # content — exclude their top-level ancestors so the preamble is the cover
    # page, not a re-render of the link list.
    toc_blocks = {id(_top_level_ancestor(el, body)) for el in toc_link_els}
    preamble_fragment = _slice_between(
        _first_body_child(body), first_start, skip=toc_blocks,
    )
    slices: list[tuple[str, str | None, str]] = []
    if preamble_fragment.strip():
        slices.append((_PREAMBLE_ANCHOR_ID, _PREAMBLE_TITLE, preamble_fragment))
    for i, (anchor_id, title, start) in enumerate(boundaries):
        next_start = boundaries[i + 1][2] if i + 1 < len(boundaries) else None
        slices.append((anchor_id, title or None, _slice_between(start, next_start)))

    sections: list[Sec2mdSection] = []
    order = 0
    for anchor_id, title, fragment in slices:
        if not fragment.strip():
            continue
        section_html = f"<html><body>{fragment}</body></html>".encode("utf-8")
        result = convert_bytes(section_html)
        if not result.chunks:
            # An anchor pointing at an empty/decorative slice carries nothing to
            # index — skip it rather than persist a content-free section row.
            continue
        sections.append(Sec2mdSection(
            anchor_id=anchor_id, title=title, order=order, result=result,
        ))
        order += 1

    # If the slices collapsed to a single (or no) real section, splitting buys
    # nothing — let the caller ingest the file whole.
    if len(sections) < _MIN_SECTIONS_TO_SPLIT:
        return []
    return sections


def _first_body_child(body):
    """The first top-level node in ``body`` (the start of the preamble slice)."""
    for node in body.children:
        if getattr(node, "name", None) is not None:
            return node
    return None


def _resolve_toc_targets(soup, body):
    """The filing's real TOC anchors, resolved to in-document targets, in
    DOCUMENT order.

    Returns ``(targets, toc_link_els)`` where each ``targets`` entry is
    ``(anchor_id, link_text, target_element)`` and ``toc_link_els`` are the TOC's
    own ``<a>`` elements (so the caller can excise the nav block from the
    preamble). The hard part is telling a genuine table of contents from the many
    other ``<a href="#id">`` links a filing carries — in-text cross-references
    ("see Item 1A"), footnote markers, and "back to top" links would each
    otherwise spawn a spurious (and out-of-order) section.

    The rule: a real TOC is a contiguous cluster of *forward* links — each link
    sits earlier in the document than the section it points at, and the cluster
    is uninterrupted in link order. We walk every internal link in link order,
    resolve it to a body target, and keep only forward links to ids not already
    listed (first link to an id wins). A "back to top" / footnote-return link
    points backward and is dropped; an in-text cross-reference to an
    already-listed section is a duplicate and is dropped; a cross-reference to an
    *un*-listed section is forward, but it sits in the body interrupted from the
    TOC by other (now-dropped) links, so it falls outside the longest contiguous
    run. We take that longest run — the TOC — then **sort it by document
    position** before returning. Sorting is what makes link order irrelevant: a
    TOC that lists sections out of document order no longer produces a slice
    whose next boundary lies earlier in the document (which would run to
    end-of-body and duplicate content).
    """
    pos = {id(el): i for i, el in enumerate(body.descendants)}

    seen: set[str] = set()
    links: list[_TocLink] = []
    for link_idx, a in enumerate(soup.find_all("a")):
        href = a.get("href") or ""
        if not href.startswith("#") or len(href) < 2:
            continue
        anchor_id = href[1:]
        if anchor_id in seen:
            continue
        target = body.find(id=anchor_id)
        if target is None or id(target) not in pos:
            continue
        link_pos = pos.get(id(a))
        target_pos = pos[id(target)]
        # Forward links only: a TOC entry sits before the content it indexes; a
        # "back to top" or footnote-return link points the other way.
        if link_pos is None or link_pos >= target_pos:
            continue
        seen.add(anchor_id)
        links.append(_TocLink(
            target_pos, anchor_id, a.get_text(strip=True), target, a, link_idx,
        ))

    run = _longest_contiguous_run(links)
    run.sort(key=lambda link: link.target_pos)
    targets = [(link.anchor_id, link.text, link.target) for link in run]
    toc_link_els = [link.el for link in run]
    return targets, toc_link_els


class _TocLink(NamedTuple):
    """One resolved, forward internal link, used while isolating the TOC cluster.

    ``target_pos`` is the link target's document position (the sort key);
    ``link_idx`` is the link's index among all ``<a>`` in the document, which
    tells the run finder where a dropped link breaks contiguity.
    """
    target_pos: int
    anchor_id: str
    text: str
    target: object
    el: object
    link_idx: int


def _longest_contiguous_run(links: list[_TocLink]) -> list[_TocLink]:
    """The longest cluster of links that is uninterrupted in link order — the TOC.

    ``links`` is the forward, deduped internal links in link order. A gap in
    their ``link_idx`` means a non-TOC link (a dropped back-to-top, or a forward
    cross-reference whose target *was* already in the TOC) sat between two kept
    links, so the run breaks there. The longest run wins; ties keep the earliest
    (the TOC sits at the top, ahead of any in-body link cluster).
    """
    if not links:
        return []
    best: list[_TocLink] = []
    current = [links[0]]
    for prev, cur in zip(links, links[1:]):
        if cur.link_idx == prev.link_idx + 1:
            current.append(cur)
        else:
            if len(current) > len(best):
                best = current
            current = [cur]
    if len(current) > len(best):
        best = current
    return best


def _top_level_ancestor(el, body):
    """The body-level block that contains ``el`` (the slice boundary unit)."""
    while el.parent is not None and el.parent is not body:
        el = el.parent
    return el


def _slice_between(start, end, *, skip: set[int] | None = None) -> str:
    """Serialize every top-level sibling from ``start`` up to (not incl) ``end``.

    Top-level blocks whose ``id()`` is in ``skip`` are omitted — used to drop the
    TOC nav block(s) from the preamble slice so navigation isn't re-indexed as
    front-matter content.
    """
    parts: list[str] = []
    node = start
    while node is not None and node is not end:
        if getattr(node, "name", None) is not None:
            if skip is None or id(node) not in skip:
                parts.append(str(node))
        node = node.find_next_sibling()
    return "".join(parts)
