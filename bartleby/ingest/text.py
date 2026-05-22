"""Plain-text fallback for `.txt` (Docling has no text reader)."""

from __future__ import annotations

from pathlib import Path

from bartleby.lib.consts import CHUNK_OVERLAP, CHUNK_SIZE


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Naive character chunker with overlap; used only for `.txt` sources.

    Inputs shorter than ``chunk_size`` come back as one chunk. Empty inputs
    yield an empty list (the caller should error out).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    stride = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start += stride
    return chunks
