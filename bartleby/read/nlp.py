import re
import unicodedata
from typing import List

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

_LIGATURES = {
    "\uFB00": "ff",
    "\uFB01": "fi",
    "\uFB02": "fl",
    "\uFB03": "ffi",
    "\uFB04": "ffl",
    "\u2010": "-",
    "\u2011": "-",
    "\u00AD": "",
    "\u00A0": " ",
    "\u200B": "",
    "\u200C": "",
    "\u200D": "",
}
_LIGATURES_TABLE = str.maketrans(_LIGATURES)

_BULLET_RE = re.compile(r"""^\s*(?:[\-\u2022\u2023\u25E6\u2043\u2219]|[A-Za-z]\)|\d+[\.\)])\s+""")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")

# ends with sentence punctuation, optionally closed by quotes/brackets
_END_SENTENCE_RE = re.compile(r"""[\.!?:;]["'”’)\]]?\s*$""")

# hyphen at end of line (word-)
_TRAILING_HYPHEN_RE = re.compile(r"[A-Za-z0-9]-\s*$")

# next line starts with alnum
_NEXT_LINE_ALNUM_RE = re.compile(r"^[A-Za-z0-9]")


def protect_match(m: re.Match) -> str:
    return m.group(0).replace("\n", "")


def clean_block_text(text: str) -> str:
    # If there's no text, return an empty string.
    if not text:
        return ""

    # Normalize and replace some shit.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_LIGATURES_TABLE)

    # Protect URLs and email addresses.
    text = _URL_RE.sub(protect_match, text)
    text = _EMAIL_RE.sub(protect_match, text)

    # Set up for line checking.
    lines = text.split("\n")
    out_lines: List[str] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]

        # Treat blank lines as paragraph breaks.
        if not line.strip():
            out_lines.append("")
            line_index += 1
            continue

        # Build a paragraph buffer.
        paragraph_buffer = line.rstrip()
        subline_index = line_index + 1
        while subline_index < len(lines):
            next_line = lines[subline_index]
            # If there's nothing next.
            if not next_line.strip():
                break
            # If there's a bullet next.
            if _BULLET_RE.match(next_line):
                break
            # De-hyphenate if previous ended with "word-" and next starts with alnum
            if _TRAILING_HYPHEN_RE.search(paragraph_buffer) and _NEXT_LINE_ALNUM_RE.match(next_line):
                paragraph_buffer = _TRAILING_HYPHEN_RE.sub("", paragraph_buffer) + next_line.lstrip()
            else:
                # If the buffer ends a sentence (optionally with quotes/brackets), stop unwrapping
                if _END_SENTENCE_RE.search(paragraph_buffer):
                    break
                paragraph_buffer += " " + next_line.lstrip()
            subline_index += 1

        out_lines.append(paragraph_buffer)
        line_index = subline_index if subline_index > line_index else line_index + 1

    text = "\n".join(out_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_body_from_pdf_page(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    if not blocks:
        return ""
    logger.debug(f"Found {len(blocks)} blocks on page.")

    blocks.sort(key=lambda b: (b[1], b[0]))

    paragraphs: List[str] = []
    for b in blocks:
        text = b[4]
        block_type = b[6] if len(b) > 6 else 0
        # Ignore non-text blocks.
        if block_type != 0:
            continue
        # Clean the text.
        cleaned = clean_block_text(text)
        if cleaned:
            paragraphs.append(cleaned)
    logger.debug(f"Found {len(paragraphs)} paragraphs on page.")

    body = "\n\n".join(paragraphs).strip()
    return body


def chunk_page_body(body: str) -> List[str]:
    from bartleby.lib.consts import CHUNK_SIZE, CHUNK_OVERLAP

    if not body or not body.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(body)
    return chunks