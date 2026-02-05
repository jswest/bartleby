import re
import unicodedata
from typing import List

import fitz
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


def _recursive_character_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Split text recursively using a list of separators, producing chunks
    up to chunk_size characters with chunk_overlap overlap.

    Mimics LangChain's RecursiveCharacterTextSplitter algorithm:
    1. Try separators in order until one splits the text.
    2. Merge small splits into chunks of up to chunk_size.
    3. Recursively split any chunk that's still too large using the next separator.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    final_chunks: List[str] = []

    # Find the appropriate separator
    separator = separators[-1]
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining_separators = []
            break
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    # Split the text
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    # Merge small splits into chunks
    current_chunk: List[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece)
        separator_len = len(separator)
        added_len = piece_len + (separator_len if current_chunk else 0)

        if current_length + added_len > chunk_size and current_chunk:
            # Flush current chunk
            merged = separator.join(current_chunk)
            if len(merged) > chunk_size and remaining_separators:
                final_chunks.extend(
                    _recursive_character_split(merged, chunk_size, chunk_overlap, remaining_separators)
                )
            elif merged.strip():
                final_chunks.append(merged.strip())

            # Keep overlap: walk backwards to find overlap content
            overlap_chunks: List[str] = []
            overlap_len = 0
            for prev in reversed(current_chunk):
                if overlap_len + len(prev) + separator_len > chunk_overlap:
                    break
                overlap_chunks.insert(0, prev)
                overlap_len += len(prev) + separator_len

            current_chunk = overlap_chunks
            current_length = sum(len(c) for c in current_chunk) + max(0, len(current_chunk) - 1) * separator_len

        current_chunk.append(piece)
        current_length += added_len

    # Flush remaining
    if current_chunk:
        merged = separator.join(current_chunk)
        if len(merged) > chunk_size and remaining_separators:
            final_chunks.extend(
                _recursive_character_split(merged, chunk_size, chunk_overlap, remaining_separators)
            )
        elif merged.strip():
            final_chunks.append(merged.strip())

    return final_chunks


def chunk_page_body(body: str) -> List[str]:
    from bartleby.lib.consts import CHUNK_SIZE, CHUNK_OVERLAP

    if not body or not body.strip():
        return []

    return _recursive_character_split(body, CHUNK_SIZE, CHUNK_OVERLAP)