"""Corpus loading and the per-doc source-text cache.

``corpus.yaml`` maps short doc-ids (``[a-z0-9-]``) to PDF file names under
``corpus/``. Extraction happens once per doc into ``sources/<doc-id>.txt`` —
the exact (truncated) text every model sees and the judge scores against.
Run and judgment records carry the text's ``source_sha`` so drift (a
pdfplumber/chunker upgrade, an edited PDF) surfaces as a hash mismatch instead
of silently mixing inputs. Delete ``sources/`` to force re-extraction.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from bartleby.benchmark.stores import BenchmarkRoot

# Production parity: the same truncation limit ingest actually uses, imported
# (not copied) so the benchmark can't silently diverge from it.
from bartleby.lib.consts import DEFAULT_MAX_SUMMARIZE_TOKENS

DOC_ID_RE = re.compile(r"^[a-z0-9-]+$")


def load_corpus(root: BenchmarkRoot) -> dict[str, Path]:
    """``corpus.yaml`` → {doc_id: pdf_path}, validating ids and existence."""
    data = yaml.safe_load(root.corpus_yaml.read_text()) or {}
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{root.corpus_yaml} must map <doc-id>: <pdf-file-name>")
    corpus: dict[str, Path] = {}
    for doc_id, file_name in data.items():
        if not DOC_ID_RE.match(str(doc_id)):
            raise SystemExit(
                f"Bad doc-id {doc_id!r} in {root.corpus_yaml} (use [a-z0-9-] only)"
            )
        pdf = root.corpus_dir / str(file_name)
        if not pdf.exists():
            raise SystemExit(f"Corpus file not found: {pdf} (doc-id {doc_id!r})")
        corpus[str(doc_id)] = pdf
    return corpus


def select_documents(corpus: dict[str, Path], documents: list[str] | None) -> dict[str, Path]:
    """Apply a ``--documents`` doc-id filter, refusing unknown ids."""
    if documents is None:
        return corpus
    unknown = [d for d in documents if d not in corpus]
    if unknown:
        raise SystemExit(
            f"Unknown doc-id(s): {', '.join(unknown)} "
            f"(corpus.yaml has: {', '.join(corpus)})"
        )
    return {d: corpus[d] for d in documents}


_ENCODING = None


def _cl100k():
    """Lazy, process-wide cl100k_base encoder."""
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    return len(_cl100k().encode(text)) if text else 0


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    tok = _cl100k()
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text, False
    return tok.decode(ids[:max_tokens]), True


def build_summary_input(pdf_path: Path) -> tuple[str, dict]:
    """Reproduce production's summary input for an image-free PDF.

    Mirrors ``scribe._parse_pdf_pdfplumber`` + ``writer.summary_input``: extract
    with the pdfplumber backend at the production thresholds, chunk each text
    page with ``chunk_text``, and join the document chunks in page/chunk order
    with blank lines. Returns ``(text, meta)``; ``meta`` flags anything that
    would make this *not* a faithful image-free reproduction (image-routed or
    sparse pages), which the caller surfaces.
    """
    from bartleby.ingest import pdfplumber as pdfplumber_pipeline
    from bartleby.ingest.text import chunk_text
    from bartleby.lib.consts import (
        DEFAULT_OCR_MIN_CONFIDENCE,
        DEFAULT_SPARSE_TEXT_THRESHOLD,
    )

    result = pdfplumber_pipeline.convert(
        pdf_path,
        sparse_text_threshold=DEFAULT_SPARSE_TEXT_THRESHOLD,
        ocr_min_confidence=DEFAULT_OCR_MIN_CONFIDENCE,
    )
    parts: list[str] = []
    image_pages: list[int] = []
    # if/elif/if is deliberate: a text page is chunked, else a sparse page would
    # route to the VLM — but *either* kind can also carry embedded images, so the
    # embedded-image check is a separate `if`, not part of the elif chain.
    for page in result.pages:
        if page.content_type is not None:
            parts.extend(chunk_text(page.text))
        elif page.page_render_png is not None:
            image_pages.append(page.page_number)  # would route to the VLM in prod
        if page.embedded_images:
            image_pages.append(page.page_number)
    meta = {
        "page_count": result.page_count,
        "image_routed_pages": sorted(set(image_pages)),
    }
    return "\n\n".join(parts), meta


def source_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SourceText:
    doc_id: str
    text: str
    sha: str
    tokens: int  # post-truncation — what the models actually see


def ensure_source(root: BenchmarkRoot, doc_id: str, pdf: Path,
                  max_tokens: int = DEFAULT_MAX_SUMMARIZE_TOKENS) -> SourceText:
    """The cached source text for a doc, extracting on first use.

    ``max_tokens`` applies only on first extraction — a cache hit serves the
    text as cached. Delete ``sources/`` to re-cut at a different limit.
    """
    cached = load_source(root, doc_id)
    if cached is not None:
        return cached
    print(f"Extracting {pdf.name} (pdfplumber + chunker)...", file=sys.stderr)
    text, meta = build_summary_input(pdf)
    if not text.strip():
        raise SystemExit(f"No extractable text in {pdf}")
    if meta["image_routed_pages"]:
        print(f"  WARNING: pages {meta['image_routed_pages']} would route to the "
              f"image/VLM pipeline — this doc is not purely image-free, so the input "
              f"omits content production would include.", file=sys.stderr)
    text, truncated = truncate_to_tokens(text, max_tokens)
    if truncated:
        print(f"  NOTE: truncated to {max_tokens} cl100k tokens.", file=sys.stderr)
    root.sources_dir.mkdir(parents=True, exist_ok=True)
    (root.sources_dir / f"{doc_id}.txt").write_text(text)
    return SourceText(doc_id, text, source_sha(text), count_tokens(text))


def load_source(root: BenchmarkRoot, doc_id: str) -> SourceText | None:
    """Read the cache without extracting (the judge-side accessor)."""
    path = root.sources_dir / f"{doc_id}.txt"
    if not path.exists():
        return None
    text = path.read_text()
    return SourceText(doc_id, text, source_sha(text), count_tokens(text))
