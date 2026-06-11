"""EDGAR full-submission (``.txt``) envelope parser.

A full submission is an SGML envelope — ``<SEC-DOCUMENT>`` / ``<SEC-HEADER>``
wrapping one or more ``<DOCUMENT>`` blocks. Each inner document is a filing
body (a 10-K/8-K, often iXBRL HTML), an exhibit, an XBRL data file, or a
uuencoded graphic. Left to the generic dispatch the whole thing lands in the
plain character chunker as raw tag soup (issue #51). This module detects the
envelope and splits it into inner documents so the scribe can route each body
to the right converter.

Scope is deliberately Phase 1: detect, split, classify. The ``<SEC-HEADER>``
metadata (CIK, ticker, form type, filing date) is *not* captured here — that
wants a schema bump and a search filter layer, and belongs to issue #10.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# A full submission opens with the SGML envelope tags; both sit at the very
# top, so sniffing a small head window is enough — and keeps us from hijacking
# ordinary ``.txt`` files that merely happen to mention SEC.
_DETECT_WINDOW_BYTES = 8192
_ENVELOPE_MARKERS = (b"<SEC-DOCUMENT>", b"<SEC-HEADER>")

# Inner blocks are SGML, not well-formed XML: the metadata tags (<TYPE>,
# <SEQUENCE>, ...) have no closing tag and their value runs to end-of-line;
# only <DOCUMENT> and <TEXT> are paired.
_DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)
_META_RE = re.compile(
    r"^<(TYPE|FILENAME)>(.*)$", re.MULTILINE
)

# Some bodies wrap their real payload in an <XBRL>/<XML> shell inside <TEXT>.
_PAYLOAD_SHELL_RE = re.compile(r"<(XBRL|XML)>(.*)</\1>", re.DOTALL | re.IGNORECASE)

# Graphic / binary exhibits arrive uuencoded inside <TEXT>.
_UUENCODE_RE = re.compile(r"^begin \d{3} ", re.MULTILINE)

_HTML_EXTS = (".htm", ".html")
_IMAGE_EXTS = (".jpg", ".jpeg", ".gif", ".png", ".bmp", ".tif", ".tiff")
_DATA_EXTS = (".xml", ".xsd", ".json", ".zip", ".pdf")
_GRAPHIC_TYPES = ("GRAPHIC", "COVER", "ZIP", "EXCEL")
_HTML_OPENERS = ("<html", "<!doctype", "<?xml", "<xbrl", "<div", "<table", "<span")


@dataclass
class InnerDocument:
    type: str | None
    filename: str | None
    text: str


def detect(path: Path) -> bool:
    """True if ``path`` opens with the EDGAR full-submission SGML envelope."""
    try:
        with path.open("rb") as f:
            head = f.read(_DETECT_WINDOW_BYTES)
    except OSError:
        return False
    return any(marker in head for marker in _ENVELOPE_MARKERS)


def parse(raw: bytes) -> list[InnerDocument]:
    """Split a full-submission envelope into its inner ``<DOCUMENT>`` blocks."""
    text = raw.decode("utf-8", errors="replace")
    docs: list[InnerDocument] = []
    for block in _DOCUMENT_RE.findall(text):
        # Parse metadata only from the portion before <TEXT> so tags that
        # happen to appear inside an HTML body can't masquerade as metadata.
        body_match = _TEXT_RE.search(block)
        header_part = block[: body_match.start()] if body_match else block
        meta = {k.lower(): v.strip() for k, v in _META_RE.findall(header_part)}

        body = body_match.group(1).strip() if body_match else ""
        shell = _PAYLOAD_SHELL_RE.search(body)
        if shell:
            body = shell.group(2).strip()

        docs.append(
            InnerDocument(
                type=meta.get("type"),
                filename=meta.get("filename"),
                text=body,
            )
        )
    return docs


def classify(doc: InnerDocument) -> str:
    """Route an inner document.

    Returns ``"html"`` (→ sec2md), ``"text"`` (→ character chunker), or
    ``"skip"`` for bodies with no usable prose: graphics, XBRL data files,
    and empties.
    """
    if not doc.text:
        return "skip"

    fname = (doc.filename or "").lower()
    dtype = (doc.type or "").upper()

    if dtype in _GRAPHIC_TYPES or fname.endswith(_IMAGE_EXTS):
        return "skip"
    if _UUENCODE_RE.search(doc.text[:200]):
        return "skip"
    # XBRL instance/schema/linkbase exhibits carry tags, not prose.
    if fname.endswith(_DATA_EXTS) or dtype.startswith("EX-101"):
        return "skip"

    if fname.endswith(_HTML_EXTS):
        return "html"
    # Fall back to a body sniff when FILENAME is missing or non-committal.
    if doc.text.lstrip().lower().startswith(_HTML_OPENERS):
        return "html"
    return "text"
