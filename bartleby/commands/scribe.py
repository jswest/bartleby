"""`bartleby scribe` — restartable ingest pipeline over a single writer.

Per source file: hash → (parse → persist → caption → summarize), where parse
produces a result and a single :class:`~bartleby.ingest.writer.Writer` drains it
to SQLite. The three post-parse units commit independently, so ingest resumes by
what's *missing*: a run that dies mid-captioning leaves the parse durable and
the next run re-captions only the images that never landed (it never re-parses,
never re-captions finished images). Deterministically-failing units are recorded
in ``failed_ingests`` and capped, not retried forever — and surfaced, so a
skipped unit never reads as a green one.

All chunk writes go through the ``bartleby.db.chunks`` typed helpers via the
Writer; image rows additionally get a join entry in ``document_images``.

PDF path is converter-aware (``pdfplumber`` default, ``docling`` opt-in).
HTML path is converter-aware too: ``docling`` is the default; ``sec2md`` is
opt-in for iXBRL EDGAR filings (sniffed per-file, non-iXBRL HTML falls back
to docling). Image files (jpg/png/etc.) go straight through the VLM
pipeline. txt/md keep the original ``convert_and_chunk`` path — except an
EDGAR full-submission ``.txt`` (detected by its SGML envelope), whose inner
documents are unwrapped and routed individually.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from bartleby.config import ensure_provider_env, load_config
from bartleby.db.chunks import ChunkInput
from bartleby.db.connection import open_db, resolve_project_name
from bartleby.ingest import edgar as edgar_pipeline
from bartleby.ingest import images as image_pipeline
from bartleby.ingest import pdfplumber as pdfplumber_pipeline
from bartleby.ingest import sec2md as sec2md_pipeline
from bartleby.ingest.chunk import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    ChunkRow,
    chunk_markdown_string,
    convert_and_chunk,
    resolve_extension,
    resolve_format_filter,
)
from bartleby.ingest.embed import embed_texts
from bartleby.ingest.summarize import count_tokens, summarize
from bartleby.ingest.text import chunk_text
from bartleby.ingest.writer import (
    MAX_INGEST_ATTEMPTS,
    ImageCaption,
    ParsedDocument,
    ParsedImage,
    PendingImage,
    Writer,
)
from bartleby.lib import console
from bartleby.lib import timing
from bartleby.lib.consts import (
    DEFAULT_HTML_CONVERTER,
    DEFAULT_OCR_MIN_CONFIDENCE,
    DEFAULT_PDF_CONVERTER,
    DEFAULT_SPARSE_TEXT_THRESHOLD,
    DEFAULT_VISION_MAX_DIMENSION,
    DEFAULT_VISION_MIN_DIMENSION,
    DOCLING_HF_REPOS,
    EMBEDDING_MODEL,
)
from bartleby.project import get_project_dir
from bartleby.providers import Provider, get_provider


# Per-file outcomes the orchestrator reports back to the run loop.
INGESTED = "ingested"          # parsed and/or resumed to completion this run
SKIPPED = "skipped"            # already complete — nothing to do
INCOMPLETE = "incomplete"      # work done, but some unit is still missing
NO_VISION_SKIP = "no_vision"   # standalone image with no vision provider (warned)

HTML_EXTENSIONS = {".html", ".htm"}


# -------------------- shared helpers --------------------


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _archive(src: Path, archive_root: Path, file_hash: str, ext: str) -> Path:
    dest_dir = archive_root / file_hash
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{file_hash}{ext}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


def _collect_files(
    paths: list[Path], only: set[str] | None = None,
) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Resolve ``paths`` to ingestible ``(file, extension)`` pairs.

    Each path is a single file or a directory walked recursively; the results
    are concatenated and de-duplicated by real path, so a file reachable from
    two supplied roots (or named twice) is collected once (issue #89).

    The resolved extension is what the file should be *treated as* — the
    filename extension when it is supported, otherwise a content-sniffed one
    (see :func:`resolve_extension`). When ``only`` is given (a set of supported,
    leading-dot extensions), files whose resolved type is not in it are dropped
    from collection — silently, since the exclusion is intentional and not the
    same as a file we couldn't identify.

    Returns ``(sources, unidentified)`` where ``unidentified`` lists directory
    entries that could not be resolved to a supported type at all; the caller
    surfaces them so they are not dropped silently.
    """
    sources: list[tuple[Path, str]] = []
    unidentified: list[Path] = []
    seen: set[Path] = set()

    def _first_seen(p: Path) -> bool:
        key = p.resolve()
        if key in seen:
            return False
        seen.add(key)
        return True

    for path in paths:
        if path.is_file():
            if not _first_seen(path):
                continue
            ext = resolve_extension(path)
            if ext is None:
                raise ValueError(f"Unsupported file type: {path.name}")
            if only is None or ext in only:
                sources.append((path, ext))
        elif path.is_dir():
            for p in sorted(path.rglob("*")):
                if not p.is_file() or not _first_seen(p):
                    continue
                ext = resolve_extension(p)
                if ext is None:
                    unidentified.append(p)
                elif only is None or ext in only:
                    sources.append((p, ext))
        else:
            raise ValueError(f"Path not found: {path}")

    return sources, unidentified


def _build_chunk_inputs(
    rows: list[ChunkRow],
    embeddings: list[list[float]],
    start_index: int = 0,
) -> list[ChunkInput]:
    return [
        ChunkInput(
            text=row.text,
            embedding=emb,
            chunk_index=start_index + i,
            section_heading=row.section_heading,
            page_number=row.page_number,
            content_type=row.content_type,
        )
        for i, (row, emb) in enumerate(zip(rows, embeddings))
    ]


def _token_count(full_text: str) -> int:
    return count_tokens(full_text) if full_text else 0


def _summary_chunks(text: str) -> list[ChunkInput]:
    """Chunk + embed a summary's markdown body into ChunkInputs (producer side)."""
    rows = chunk_markdown_string(text)
    if not rows:
        return []
    embeddings = embed_texts([r.text for r in rows])
    return _build_chunk_inputs(rows, embeddings)


# -------------------- image parse + caption --------------------


class _ImageRoute:
    """One image (embedded or page-render) ready for the pipeline."""

    def __init__(
        self,
        bytes_: bytes,
        page_number: int | None,
        image_index_on_page: int,
    ):
        self.bytes_ = bytes_
        self.page_number = page_number              # None for standalone files
        self.image_index_on_page = image_index_on_page  # 0 for page-renders/standalone


def _parse_image_routes(
    routes: list[_ImageRoute],
    *,
    archive_root: Path,
    vision_enabled: bool,
    vision_max_dimension: int,
    vision_min_dimension: int,
) -> list[ParsedImage]:
    """Scale + archive each route into a ParsedImage (no VLM, no DB).

    Sub-minimum images are dropped here, exactly as before: VLM image processors
    crash on sub-patch-size edges and such slivers carry no describable content.
    With no vision provider there's nothing to caption, so routes produce no
    rows (a warning, then skipped).
    """
    if not routes:
        return []
    if not vision_enabled:
        console.warn(
            f"Skipping {len(routes)} image(s) — no vision provider configured."
        )
        return []
    parsed: list[ParsedImage] = []
    for route in routes:
        prepared = image_pipeline.prepare_image(
            route.bytes_, max_dimension=vision_max_dimension,
        )
        if image_pipeline.is_below_vlm_minimum(
            prepared, min_dimension=vision_min_dimension
        ):
            page_str = f" (page {route.page_number})" if route.page_number else ""
            console.warn(
                f"Skipping image{page_str} — {prepared.width}x{prepared.height}px "
                f"is below the {vision_min_dimension}px vision minimum."
            )
            continue
        archived = image_pipeline.archive_image(prepared, archive_root)
        parsed.append(ParsedImage(
            hash=prepared.hash,
            archive_path=archived,
            width=prepared.width,
            height=prepared.height,
            page_number=route.page_number,
            image_index_on_page=route.image_index_on_page,
        ))
    return parsed


def _caption_image(
    pending: PendingImage,
    jpeg_bytes: bytes,
    *,
    vision_provider: Provider,
    vision_model: str,
) -> ImageCaption:
    """Run the §7 image pipeline on one already-recorded image row.

    Decides text-image vs scene-image (Tesseract first; VLM only for scenes),
    producing the analysis JSON and at most one searchable chunk. The prepared
    JPEG is reloaded from the archive by the caller.
    """
    prepared = image_pipeline.PreparedImage(
        hash=pending.file_hash,
        jpeg_bytes=jpeg_bytes,
        width=pending.width,
        height=pending.height,
    )
    analysis = image_pipeline.analyze(
        vision_provider, prepared, model=vision_model,
    )
    return ImageCaption(
        image_id=pending.image_id,
        analysis_json=analysis.model_dump_json(),
        analysis_model=vision_model,
        chunks=image_pipeline.analysis_to_chunk_inputs(analysis),
    )


# -------------------- per-type parse (no DB writes) --------------------


def _parse_text_document(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """txt / html / md via the existing convert_and_chunk path."""
    if on_stage is not None:
        on_stage("extracting")
    result = convert_and_chunk(archived)
    chunks: list[ChunkInput] = []
    if result.chunks:
        embeddings = embed_texts([row.text for row in result.chunks])
        chunks = _build_chunk_inputs(result.chunks, embeddings)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=result.page_count, token_count=_token_count(result.full_text),
        document_chunks=chunks, images=[],
    )


def _sec2md_chunk_to_row(chunk, *, fallback_heading: str | None = None) -> ChunkRow:
    """Map a ``Sec2mdChunk`` to a ``ChunkRow``, falling back to ``fallback_heading``
    when sec2md didn't detect a section header for the chunk."""
    return ChunkRow(
        text=chunk.text,
        section_heading=chunk.section_heading or fallback_heading,
        content_type=chunk.content_type,
        page_number=chunk.page_number,
    )


def _parse_html_sec2md(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """iXBRL EDGAR filing via sec2md."""
    if on_stage is not None:
        on_stage("extracting")
    result = sec2md_pipeline.convert(archived)
    chunks: list[ChunkInput] = []
    if result.chunks:
        if on_stage is not None:
            on_stage("embedding")
        rows = [_sec2md_chunk_to_row(c) for c in result.chunks]
        embeddings = embed_texts([r.text for r in rows])
        chunks = _build_chunk_inputs(rows, embeddings)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=result.page_count, token_count=_token_count(result.full_text),
        document_chunks=chunks, images=[],
    )


def _parse_edgar_submission(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """EDGAR full-submission `.txt`: unwrap the SGML envelope and route each
    inner document to its converter, landing as one document row.

    Inner HTML/iXBRL bodies always go through sec2md (the only thing that reads
    SEC HTML — there is no working fallback here, so the dependency is hard);
    plain-text bodies go through the character chunker; graphics and XBRL data
    files are skipped. Origin of each chunk is recorded in ``section_heading``.
    """
    if on_stage is not None:
        on_stage("extracting")
    inner_documents = edgar_pipeline.parse(archived.read_bytes())

    rows: list[ChunkRow] = []
    full_text_parts: list[str] = []
    for doc in inner_documents:
        label = doc.type or doc.filename or "document"
        kind = edgar_pipeline.classify(doc)
        if kind == "skip":
            console.warn(f"{file_name}: skipping inner document {label}.")
            continue
        if kind == "html":
            result = sec2md_pipeline.convert_bytes(doc.text.encode("utf-8"))
            full_text_parts.append(result.full_text)
            rows.extend(
                _sec2md_chunk_to_row(c, fallback_heading=label)
                for c in result.chunks
            )
        else:  # "text"
            full_text_parts.append(doc.text)
            rows.extend(
                ChunkRow(text=t, section_heading=label, content_type=None)
                for t in chunk_text(doc.text)
            )

    full_text = "\n\f\n".join(part for part in full_text_parts if part)
    chunks: list[ChunkInput] = []
    if rows:
        if on_stage is not None:
            on_stage("embedding")
        embeddings = embed_texts([r.text for r in rows])
        chunks = _build_chunk_inputs(rows, embeddings)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=None, token_count=_token_count(full_text),
        document_chunks=chunks, images=[],
    )


def _parse_pdf_pdfplumber(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    archive_root: Path,
    sparse_text_threshold: int,
    ocr_min_confidence: int,
    vision_enabled: bool,
    vision_max_dimension: int,
    vision_min_dimension: int,
    on_page_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_page_count: Callable[[int | None], None] | None = None,
) -> ParsedDocument:
    if on_stage is not None:
        on_stage("extracting")
    result = pdfplumber_pipeline.convert(
        archived,
        sparse_text_threshold=sparse_text_threshold,
        ocr_min_confidence=ocr_min_confidence,
        on_progress=on_page_progress,
    )
    if on_page_count is not None:
        on_page_count(result.page_count)

    doc_rows: list[ChunkRow] = []
    image_routes: list[_ImageRoute] = []
    for page in result.pages:
        if page.content_type is not None:
            for piece in chunk_text(page.text):
                doc_rows.append(ChunkRow(
                    text=piece,
                    section_heading=None,
                    content_type=page.content_type,
                    page_number=page.page_number,
                ))
        elif page.page_render_png is not None:
            # Sparse page where OCR didn't clear the bar — fall back to VLM.
            # The caption stage runs its own Tesseract pre-pass on the archived
            # render (parse no longer forwards page-level OCR across the queue).
            image_routes.append(_ImageRoute(
                bytes_=page.page_render_png,
                page_number=page.page_number,
                image_index_on_page=0,
            ))

        for emb in page.embedded_images:
            image_routes.append(_ImageRoute(
                bytes_=emb.png_bytes,
                page_number=page.page_number,
                image_index_on_page=emb.image_index_on_page,
            ))

    chunks: list[ChunkInput] = []
    if doc_rows:
        if on_stage is not None:
            on_stage("embedding")
        embeddings = embed_texts([r.text for r in doc_rows])
        chunks = _build_chunk_inputs(doc_rows, embeddings)

    images = _parse_image_routes(
        image_routes, archive_root=archive_root, vision_enabled=vision_enabled,
        vision_max_dimension=vision_max_dimension,
        vision_min_dimension=vision_min_dimension,
    )
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=result.page_count, token_count=_token_count(result.full_text),
        document_chunks=chunks, images=images,
    )


def _parse_pdf_docling(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    archive_root: Path,
    vision_enabled: bool,
    vision_max_dimension: int,
    vision_min_dimension: int,
    on_stage: Callable[[str], None] | None = None,
    on_page_count: Callable[[int | None], None] | None = None,
) -> ParsedDocument:
    from bartleby.ingest import docling as docling_pipeline

    if on_stage is not None:
        on_stage("extracting")
    docling_result = docling_pipeline.convert(
        archived, extract_images=vision_enabled,
    )
    if on_page_count is not None:
        on_page_count(docling_result.page_count)

    chunks: list[ChunkInput] = []
    if docling_result.chunks:
        if on_stage is not None:
            on_stage("embedding")
        rows = [
            ChunkRow(text=c.text, section_heading=c.section_heading,
                     content_type=c.content_type, page_number=c.page_number)
            for c in docling_result.chunks
        ]
        embeddings = embed_texts([r.text for r in rows])
        chunks = _build_chunk_inputs(rows, embeddings)

    # Embedded images come out of the same docling pass (no second parse).
    image_routes: list[_ImageRoute] = []
    if vision_enabled and docling_result.images:
        image_routes = [
            _ImageRoute(
                bytes_=img.png_bytes,
                page_number=img.page_number,
                image_index_on_page=img.image_index_on_page,
            )
            for img in docling_result.images
        ]
    images = _parse_image_routes(
        image_routes, archive_root=archive_root, vision_enabled=vision_enabled,
        vision_max_dimension=vision_max_dimension,
        vision_min_dimension=vision_min_dimension,
    )
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=docling_result.page_count,
        token_count=_token_count(docling_result.full_text),
        document_chunks=chunks, images=images,
    )


def _parse_image_file(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    archive_root: Path,
    vision_max_dimension: int,
    vision_min_dimension: int,
) -> ParsedDocument:
    """A standalone image: one route, no document chunks. The summarizer (if
    enabled) builds its input from the image chunk once captioned."""
    route = _ImageRoute(
        bytes_=archived.read_bytes(), page_number=None, image_index_on_page=0,
    )
    images = _parse_image_routes(
        [route], archive_root=archive_root, vision_enabled=True,
        vision_max_dimension=vision_max_dimension,
        vision_min_dimension=vision_min_dimension,
    )
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=None, token_count=0, document_chunks=[], images=images,
    )


def _parse_document(
    archived: Path,
    ext: str,
    *,
    file_hash: str,
    file_name: str,
    pdf_converter: str,
    html_converter: str,
    sparse_text_threshold: int,
    ocr_min_confidence: int,
    vision_enabled: bool,
    vision_max_dimension: int,
    vision_min_dimension: int,
    archive_root: Path,
    on_page_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_page_count: Callable[[int | None], None] | None = None,
) -> ParsedDocument:
    """Route an archived file to its converter and return a parsed result."""
    if ext in IMAGE_EXTENSIONS:
        return _parse_image_file(
            archived, file_hash=file_hash, file_name=file_name,
            archive_root=archive_root,
            vision_max_dimension=vision_max_dimension,
            vision_min_dimension=vision_min_dimension,
        )
    if ext in PDF_EXTENSIONS:
        if pdf_converter == "docling":
            return _parse_pdf_docling(
                archived, file_hash=file_hash, file_name=file_name,
                archive_root=archive_root, vision_enabled=vision_enabled,
                vision_max_dimension=vision_max_dimension,
                vision_min_dimension=vision_min_dimension,
                on_stage=on_stage, on_page_count=on_page_count,
            )
        return _parse_pdf_pdfplumber(
            archived, file_hash=file_hash, file_name=file_name,
            archive_root=archive_root,
            sparse_text_threshold=sparse_text_threshold,
            ocr_min_confidence=ocr_min_confidence,
            vision_enabled=vision_enabled,
            vision_max_dimension=vision_max_dimension,
            vision_min_dimension=vision_min_dimension,
            on_page_progress=on_page_progress,
            on_stage=on_stage, on_page_count=on_page_count,
        )
    if edgar_pipeline.detect(archived):
        # EDGAR full-submission SGML envelope — detected by content, so a `.txt`
        # wrapper is caught here instead of falling to the character chunker.
        return _parse_edgar_submission(
            archived, file_hash=file_hash, file_name=file_name, on_stage=on_stage,
        )
    if (
        ext in HTML_EXTENSIONS
        and html_converter == "sec2md"
        and sec2md_pipeline.is_ixbrl(archived)
    ):
        return _parse_html_sec2md(
            archived, file_hash=file_hash, file_name=file_name, on_stage=on_stage,
        )
    return _parse_text_document(
        archived, file_hash=file_hash, file_name=file_name, on_stage=on_stage,
    )


# -------------------- per-file orchestration --------------------


def _is_complete(writer: Writer, document_id: int, *, summaries_enabled: bool) -> bool:
    """Per-unit completeness: parsed ∧ every image captioned ∧ summary settled.

    "Summary settled" means a summary row exists, summaries are disabled, or the
    document has no summarizable content (0 chunks → nothing to summarize).
    """
    if writer.uncaptioned_images(document_id):
        return False
    if (
        summaries_enabled
        and not writer.summary_exists(document_id)
        and writer.summarizable_chunk_count(document_id) > 0
    ):
        return False
    return True


def _caption_missing(
    writer: Writer,
    document_id: int,
    file_name: str,
    *,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_enabled: bool,
    on_stage: Callable[[str], None] | None,
    on_image_progress: Callable[[int, int], None] | None,
) -> bool:
    """Caption every uncaptioned image row of a document. Returns True if any
    image is still uncaptioned afterwards (failed this run or capped)."""
    pending = writer.uncaptioned_images(document_id)
    if not pending:
        return False
    if not vision_enabled:
        console.warn(
            f"Skipping {len(pending)} uncaptioned image(s) in {file_name} — "
            f"no vision provider configured."
        )
        return True

    if on_stage is not None:
        on_stage("analyzing images")
    total = len(pending)
    if on_image_progress is not None:
        on_image_progress(0, total)

    incomplete = False
    for i, pi in enumerate(pending, start=1):
        page_str = f" (page {pi.page_number})" if pi.page_number else ""
        try:
            if writer.is_capped(pi.file_hash, "caption"):
                incomplete = True
                console.warn(
                    f"{file_name}: skipping image{page_str} — failed "
                    f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
                )
                continue
            jpeg = Path(pi.file_path).read_bytes()
            caption = _caption_image(
                pi, jpeg, vision_provider=vision_provider,
                vision_model=vision_model,
            )
            writer.persist_caption(caption)
            writer.clear_failure(pi.file_hash, "caption")
        except Exception as e:
            incomplete = True
            console.warn(f"Image analysis failed{page_str}: {e}")
            writer.record_failure(pi.file_hash, file_name, "caption", e)
        finally:
            if on_image_progress is not None:
                on_image_progress(i, total)
    return incomplete


def _summarize_if_missing(
    writer: Writer,
    document_id: int,
    file_hash: str,
    file_name: str,
    *,
    summaries_enabled: bool,
    llm_provider: Provider | None,
    llm_model: str | None,
    temperature: float,
    max_summarize_tokens: int,
    on_stage: Callable[[str], None] | None,
) -> bool:
    """Summarize a document if it has none yet. Returns True if a summary was
    owed but did not land (failed this run or capped)."""
    if not summaries_enabled or writer.summary_exists(document_id):
        return False
    if writer.summarizable_chunk_count(document_id) == 0:
        # No indexed chunks → nothing real to summarize. Handing the model trace
        # text (page labels, form-feed separators from an image-only/sub-threshold
        # doc) makes it confabulate (issue #80). Counts as settled; surface it.
        console.warn(
            f"{file_name}: no summary — no extractable content (0 chunks). "
            f"The document may be image-only and need OCR/re-ingest."
        )
        return False
    if writer.is_capped(file_hash, "summary"):
        console.warn(
            f"{file_name}: skipping summary — failed {MAX_INGEST_ATTEMPTS}× "
            f"already; not retrying."
        )
        return True
    try:
        if on_stage is not None:
            on_stage("summarizing")
        result = summarize(
            writer.summary_input(document_id),
            provider=llm_provider, model=llm_model,
            temperature=temperature, max_summarize_tokens=max_summarize_tokens,
        )
        writer.persist_summary(document_id, result, _summary_chunks(result.text))
        writer.clear_failure(file_hash, "summary")
        return False
    except Exception as e:
        console.warn(f"{file_name}: summary failed: {e}")
        writer.record_failure(file_hash, file_name, "summary", e)
        return True


def _process_one(
    writer: Writer,
    path: Path,
    ext: str,
    archive_root: Path,
    *,
    pdf_converter: str,
    html_converter: str,
    sparse_text_threshold: int,
    ocr_min_confidence: int,
    vision_max_dimension: int,
    vision_min_dimension: int,
    llm_provider: Provider | None,
    llm_model: str | None,
    temperature: float,
    max_summarize_tokens: int,
    vision_provider: Provider | None,
    vision_model: str | None,
    on_page_progress: Callable[[int, int], None] | None = None,
    on_image_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_page_count: Callable[[int | None], None] | None = None,
) -> str:
    """Ingest one file, resuming only the units that are missing.

    Returns one of the module-level outcome constants. A parse failure is
    recorded and re-raised (the run loop reports it); caption and summary
    failures are recorded but not raised — they leave the document incomplete,
    to be resumed on a later run until they succeed or hit the retry cap.
    """
    file_hash = _hash_file(path)
    summaries_enabled = llm_provider is not None and bool(llm_model)
    vision_enabled = vision_provider is not None and bool(vision_model)

    document_id = writer.document_id_for(file_hash)
    if document_id is not None and _is_complete(
        writer, document_id, summaries_enabled=summaries_enabled,
    ):
        return SKIPPED

    # PARSE — only when the document hasn't been parsed yet. The parse result
    # commits as one transaction, so this never re-parses an existing document.
    if document_id is None:
        if ext in IMAGE_EXTENSIONS and not vision_enabled:
            console.warn(
                f"{path.name}: skipping image (no vision provider configured)."
            )
            return NO_VISION_SKIP
        archived = _archive(path, archive_root, file_hash, ext)
        try:
            parsed = _parse_document(
                archived, ext, file_hash=file_hash, file_name=path.name,
                pdf_converter=pdf_converter, html_converter=html_converter,
                sparse_text_threshold=sparse_text_threshold,
                ocr_min_confidence=ocr_min_confidence,
                vision_enabled=vision_enabled,
                vision_max_dimension=vision_max_dimension,
                vision_min_dimension=vision_min_dimension,
                archive_root=archive_root,
                on_page_progress=on_page_progress,
                on_stage=on_stage, on_page_count=on_page_count,
            )
        except Exception as e:
            writer.record_failure(file_hash, path.name, "parse", e)
            raise
        document_id = writer.persist_parse(parsed)
        writer.clear_failure(file_hash, "parse")

    incomplete = _caption_missing(
        writer, document_id, path.name,
        vision_provider=vision_provider, vision_model=vision_model,
        vision_enabled=vision_enabled,
        on_stage=on_stage, on_image_progress=on_image_progress,
    )
    incomplete = _summarize_if_missing(
        writer, document_id, file_hash, path.name,
        summaries_enabled=summaries_enabled,
        llm_provider=llm_provider, llm_model=llm_model,
        temperature=temperature, max_summarize_tokens=max_summarize_tokens,
        on_stage=on_stage,
    ) or incomplete

    return INCOMPLETE if incomplete else INGESTED


# -------------------- resolution / entry --------------------


def _resolve_llm_provider(
    config: dict,
    *,
    provider_override: str | None,
    model_override: str | None,
) -> tuple[Provider | None, str | None]:
    if config.get("summary_depth", "none") != "one-shot":
        return None, None

    name = provider_override or config.get("provider")
    model = model_override or config.get("model")
    if not name or not model:
        console.warn(
            "summary_depth=one-shot but no provider/model configured; "
            "skipping summaries."
        )
        return None, None

    ensure_provider_env(name, config)
    return get_provider(name, ollama_base_url=config.get("ollama_base_url")), model


def _resolve_vision_provider(
    config: dict,
) -> tuple[Provider | None, str | None]:
    name = config.get("vision_provider")
    model = config.get("vision_model")
    if not name or not model:
        return None, None
    ensure_provider_env(name, config)
    return get_provider(name, ollama_base_url=config.get("ollama_base_url")), model


def _required_hf_models(pdf_converter: str, html_converter: str) -> tuple[str, ...]:
    """HF repos this ingest run will load, used to gate offline mode (#88).

    The embedding model is always needed; docling's layout/table models only
    when a docling converter is active for PDFs or HTML. The gate stays online
    until every model here is cached, so lazy downloads succeed.
    """
    models = [EMBEDDING_MODEL]
    if "docling" in (pdf_converter, html_converter):
        models.extend(DOCLING_HF_REPOS)
    return tuple(models)


def _report_failures(writer: Writer) -> None:
    """Surface every still-unresolved ingest unit so a skipped one never reads
    as a green run. Capped units won't be retried; the rest resume next run."""
    failures = writer.failures()
    if not failures:
        return
    capped = sum(1 for f in failures if f.capped)
    console.warn(
        f"{len(failures)} ingest unit(s) did not complete "
        f"({capped} capped after {MAX_INGEST_ATTEMPTS} attempts, "
        f"the rest will retry on the next run):"
    )
    DISPLAY_LIMIT = 10
    for f in failures[:DISPLAY_LIMIT]:
        flag = (
            "capped — not retried" if f.capped
            else f"will retry (attempt {f.attempts}/{MAX_INGEST_ATTEMPTS})"
        )
        console.warn(f"  [{f.stage}] {f.file_name}: {f.error} — {flag}")
    if len(failures) > DISPLAY_LIMIT:
        console.warn(
            f"  … and {len(failures) - DISPLAY_LIMIT} more "
            f"(see `bartleby project info`)"
        )


def main(
    *,
    project: str | None,
    files: str | Path | list[str | Path],
    only: list[str] | None = None,
    model: str | None = None,
    provider: str | None = None,
    pdf_converter: str | None = None,
    html_converter: str | None = None,
    verbose: bool = False,
    timings: bool = False,
) -> None:
    # Resolve converters before quietening third parties: which models the run
    # needs (and thus whether offline mode is safe) depends on them. Reading
    # config and arg strings imports no ML libs, so env vars still land before
    # the libraries do.
    config = load_config()
    pdf_converter_name = (
        pdf_converter or config.get("pdf_converter", DEFAULT_PDF_CONVERTER)
    ).lower()
    if pdf_converter_name not in ("pdfplumber", "docling"):
        raise ValueError(
            f"Unknown pdf_converter {pdf_converter_name!r}; "
            f"expected 'pdfplumber' or 'docling'."
        )
    html_converter_name = (
        html_converter or config.get("html_converter", DEFAULT_HTML_CONVERTER)
    ).lower()
    if html_converter_name not in ("docling", "sec2md"):
        raise ValueError(
            f"Unknown html_converter {html_converter_name!r}; "
            f"expected 'docling' or 'sec2md'."
        )

    from bartleby.lib.quiet import setup_quiet_third_party
    setup_quiet_third_party(
        verbose=verbose,
        required_models=_required_hf_models(pdf_converter_name, html_converter_name),
    )

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    project_name = resolve_project_name(project)

    llm_provider, llm_model = _resolve_llm_provider(
        config, provider_override=provider, model_override=model,
    )
    vision_provider, vision_model = _resolve_vision_provider(config)

    # `files` is one path or many (CLI passes a list via nargs="+"); normalize
    # so collection always walks a list. `--only` names are comma-splittable and
    # repeatable, resolved to the set of extensions to keep.
    file_args = [files] if isinstance(files, (str, Path)) else files
    only_filter: set[str] | None = None
    if only:
        names = [n for group in only for n in group.split(",") if n.strip()]
        if names:
            only_filter = resolve_format_filter(names)

    sources, unidentified = _collect_files(
        [Path(f) for f in file_args], only_filter,
    )
    if unidentified:
        UNIDENTIFIED_DISPLAY_LIMIT = 10
        console.warn(
            f"Skipping {len(unidentified)} file(s) whose type could not be "
            f"identified (no usable extension and content sniffing failed):"
        )
        for p in unidentified[:UNIDENTIFIED_DISPLAY_LIMIT]:
            console.warn(f"  - {p.name}")
        if len(unidentified) > UNIDENTIFIED_DISPLAY_LIMIT:
            console.warn(
                f"  … and {len(unidentified) - UNIDENTIFIED_DISPLAY_LIMIT} more"
            )
    if not sources:
        console.warn("No supported files found.")
        return

    archive_root = get_project_dir(project_name) / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    console.big(f"Ingesting {len(sources)} file(s) into project '{project_name}'")

    conn = open_db(project_name)
    writer = Writer(conn)
    try:
        temperature = float(config.get("temperature", 0))
        max_summarize_tokens = int(config.get("max_summarize_tokens", 50_000))
        sparse_text_threshold = int(
            config.get("sparse_text_threshold", DEFAULT_SPARSE_TEXT_THRESHOLD)
        )
        ocr_min_confidence = int(
            config.get("ocr_min_confidence", DEFAULT_OCR_MIN_CONFIDENCE)
        )
        vision_max_dimension = int(
            config.get("vision_max_dimension", DEFAULT_VISION_MAX_DIMENSION)
        )
        vision_min_dimension = int(
            config.get("vision_min_dimension", DEFAULT_VISION_MIN_DIMENSION)
        )

        # Share the console module's Rich Console so messages printed via
        # console.info/error during the bar's lifetime are inserted above
        # the Live display rather than colliding with it. That Console is
        # on stderr, which also keeps stdout-redirect captures around
        # Docling/RapidOCR from breaking the bar.
        with Progress(
            TextColumn("[bold]{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console.get_console(),
        ) as bar:
            task = bar.add_task("files", total=len(sources), label="Ingesting")
            sub = bar.add_task(
                "sub", total=None, label="", visible=False,
            )

            def _phase_callback(phase: str) -> Callable[[int, int], None]:
                def cb(done: int, total: int) -> None:
                    if done == 0:
                        bar.reset(
                            sub, total=total, completed=0,
                            visible=True, label=phase,
                        )
                    else:
                        bar.update(sub, completed=done, total=total)
                return cb

            SKIP_DISPLAY_LIMIT = 3
            skipped_count = 0
            incomplete_count = 0

            # --timings: collect per-doc per-stage wall-clock and aggregate it
            # after the loop. Off by default — `timer` stays None and the loop
            # runs unchanged. `run_start` brackets the whole loop so docs/sec
            # and pages/sec count inter-doc overhead, not just stage time.
            timing_records: list[timing.DocTiming] = []
            run_start = time.perf_counter() if timings else None

            for src, src_ext in sources:
                # Render the main row from the active stage + page count. The
                # filename is truncated so a long name can't squeeze the bar
                # (issue #85); the page count is appended once extraction knows
                # it, for whichever converter ran.
                state = {"stage": "starting", "pages": None}
                timer = timing.StageTimer() if timings else None

                def _render(_src=src, _state=state) -> None:
                    name = console.truncate_filename(_src.name)
                    pages = _state["pages"]
                    page_str = (
                        f" · {pages} page{'' if pages == 1 else 's'}"
                        if pages else ""
                    )
                    bar.update(
                        task,
                        label=f"Ingesting {name}{page_str} · {_state['stage']}",
                    )

                def _set_stage(
                    stage: str, _state=state, _render=_render, _timer=timer,
                ) -> None:
                    _state["stage"] = stage
                    _render()
                    if _timer is not None:
                        _timer.mark(stage)

                def _set_page_count(count: int | None, _state=state, _render=_render) -> None:
                    _state["pages"] = count
                    _render()

                _render()
                try:
                    outcome = _process_one(
                        writer, src, src_ext, archive_root,
                        pdf_converter=pdf_converter_name,
                        html_converter=html_converter_name,
                        sparse_text_threshold=sparse_text_threshold,
                        ocr_min_confidence=ocr_min_confidence,
                        vision_max_dimension=vision_max_dimension,
                        vision_min_dimension=vision_min_dimension,
                        llm_provider=llm_provider, llm_model=llm_model,
                        temperature=temperature,
                        max_summarize_tokens=max_summarize_tokens,
                        vision_provider=vision_provider,
                        vision_model=vision_model,
                        on_page_progress=_phase_callback("  pages"),
                        on_image_progress=_phase_callback("  images"),
                        on_stage=_set_stage,
                        on_page_count=_set_page_count,
                    )
                    if outcome == SKIPPED:
                        skipped_count += 1
                        if skipped_count <= SKIP_DISPLAY_LIMIT:
                            console.info(
                                f"Skipping {src.name} (already ingested)"
                            )
                    elif outcome in (INGESTED, INCOMPLETE):
                        # A document that did real work this run (whether it
                        # finished or is resumable) — count incompletes, and
                        # record its per-stage timing when --timings is on.
                        if outcome == INCOMPLETE:
                            incomplete_count += 1
                        if timer is not None:
                            timer.finish()
                            timing_records.append(timing.DocTiming(
                                page_count=state["pages"],
                                stages=dict(timer.totals),
                            ))
                            console.info(timing.render_doc_line(
                                src.name, timer.total, timer.totals,
                            ))
                except Exception as e:
                    from bartleby.lib.quiet import OFFLINE_HINT, offline_blocked
                    message = f"Failed: {src.name}: {e}"
                    if offline_blocked(e):
                        message += f"\n  {OFFLINE_HINT}"
                    console.error(message)
                    if verbose:
                        logger.exception(e)
                finally:
                    bar.update(sub, visible=False)
                    bar.advance(task)

            extra_skipped = skipped_count - SKIP_DISPLAY_LIMIT
            if extra_skipped > 0:
                console.info(
                    f"… and {extra_skipped} more file(s) skipped as already ingested"
                )
            if incomplete_count:
                console.warn(
                    f"{incomplete_count} file(s) ingested but incomplete — "
                    f"some unit(s) still missing (see below)."
                )

        _report_failures(writer)
    finally:
        conn.close()

    if timings:
        # Human breakdown to stderr; machine summary to stdout so a benchmark
        # run is captured with `bartleby scribe --files <sample> --timings >
        # bench.json` while the bar and status text stay on stderr.
        agg = timing.aggregate(timing_records, time.perf_counter() - run_start)
        console.big("Timing summary")
        for line in timing.render_summary(agg):
            console.info(line)
        print(json.dumps(agg), flush=True)

    console.complete("Done.")
