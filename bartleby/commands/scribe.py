"""`bartleby scribe` — restartable ingest pipeline over a single writer.

Per source file: hash → (parse → persist → caption), where parse produces a
result and a single :class:`~bartleby.ingest.writer.Writer` drains it to SQLite;
summarization then runs as its own pass over every document still owing one,
once the parse/caption drain is done — kept off the parse path so it can't
throttle the parse workers (issue #167). Every unit commits independently, so
ingest resumes by what's *missing*: a run that dies mid-captioning leaves the
parse durable and the next run re-captions only the images that never landed
(it never re-parses, never re-captions finished images) and then summarizes
whatever lacks a summary. Deterministically-failing units are recorded in
``failed_ingests`` and capped, not retried forever — and surfaced, so a skipped
unit never reads as a green one.

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
import itertools
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
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

from bartleby.config import (
    config_drift,
    ensure_provider_env,
    load_config,
    redact_config,
)
from bartleby.db.chunks import ChunkInput
from bartleby.db.connection import open_db, resolve_project_name
from bartleby.ingest import edgar as edgar_pipeline
from bartleby.ingest import images as image_pipeline
from bartleby.ingest import pdfplumber as pdfplumber_pipeline
from bartleby.ingest import pool
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
    PendingSummary,
    Writer,
)
from bartleby.lib import console
from bartleby.lib import timing
from bartleby.lib.consts import (
    DEFAULT_CAPTION_WORKERS,
    DEFAULT_HTML_CONVERTER,
    DEFAULT_OCR_MIN_CONFIDENCE,
    DEFAULT_PDF_CONVERTER,
    DEFAULT_SPARSE_TEXT_THRESHOLD,
    DEFAULT_SUMMARIZE_WORKERS,
    DEFAULT_VISION_MAX_DIMENSION,
    DEFAULT_VISION_MIN_DIMENSION,
    DOCLING_HF_REPOS,
    EMBEDDING_MODEL,
    PER_WORKER_GB,
)
from bartleby.project import get_project_dir
from bartleby.providers import Provider, get_provider


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


def _analyze_image(
    pending: PendingImage,
    *,
    vision_provider: Provider,
    vision_model: str,
) -> image_pipeline.ImageAnalysis:
    """Run the §7 image pipeline (Tesseract → VLM) on one recorded image row.

    Reloads the prepared JPEG from the archive and classifies text-image vs
    scene-image (VLM only for scenes). Pure analysis: no embedding, no DB — the
    OCR subprocess and VLM network call both release the GIL, so this is the
    half of captioning that runs off the writer thread in the caption pool.
    """
    prepared = image_pipeline.PreparedImage(
        hash=pending.file_hash,
        jpeg_bytes=Path(pending.file_path).read_bytes(),
        width=pending.width,
        height=pending.height,
    )
    return image_pipeline.analyze(vision_provider, prepared, model=vision_model)


def _caption_from_analysis(
    pending: PendingImage,
    analysis: image_pipeline.ImageAnalysis,
    vision_model: str,
) -> ImageCaption:
    """Package an analysis into a writer-ready caption, embedding its chunk.

    Runs on the writer thread: ``analysis_to_chunk_inputs`` embeds via the one
    per-process SentenceTransformer, which isn't safe to call from several
    caption threads at once — so embedding stays here, next to the DB write.
    """
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
    on_stage: Callable[[str], None] | None = None,
) -> ParsedDocument:
    if on_stage is not None:
        on_stage("extracting")
    result = pdfplumber_pipeline.convert(
        archived,
        sparse_text_threshold=sparse_text_threshold,
        ocr_min_confidence=ocr_min_confidence,
    )

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
) -> ParsedDocument:
    from bartleby.ingest import docling as docling_pipeline

    if on_stage is not None:
        on_stage("extracting")
    docling_result = docling_pipeline.convert(
        archived, extract_images=vision_enabled,
    )

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
    on_stage: Callable[[str], None] | None = None,
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
                on_stage=on_stage,
            )
        return _parse_pdf_pdfplumber(
            archived, file_hash=file_hash, file_name=file_name,
            archive_root=archive_root,
            sparse_text_threshold=sparse_text_threshold,
            ocr_min_confidence=ocr_min_confidence,
            vision_enabled=vision_enabled,
            vision_max_dimension=vision_max_dimension,
            vision_min_dimension=vision_min_dimension,
            on_stage=on_stage,
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


def _is_complete(writer: Writer, document_id: int) -> bool:
    """Per-unit drain completeness: parsed ∧ every image captioned.

    Summaries are settled by their own pass (:func:`_summarize_all`), not
    here — a document missing only its summary is "complete" for the
    parse/caption drain and is picked up later from the DB by that pass.
    """
    return not writer.uncaptioned_images(document_id)


@dataclass
class _DocUnit:
    """A document parsed + persisted this run (or resumed from an earlier one),
    carried through the caption and summarize phases. ``stages`` accumulates
    per-stage seconds only under ``--timings`` — it's None on the fast path."""
    document_id: int
    file_name: str
    file_hash: str
    page_count: int | None = None
    stages: dict[str, float] | None = None


def _caption_all(
    writer: Writer,
    units: list[_DocUnit],
    *,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_enabled: bool,
    caption_workers: int,
    timings: bool,
    on_progress: Callable[[int, int], None] | None,
) -> None:
    """Phase 2: caption every still-uncaptioned image across all parsed units.

    Captioning is decoupled from parse (#166): parse only records image rows
    (deduped by byte-hash, ``analysis_json IS NULL``); this stage analyzes them
    and joins the captions back through the single Writer. Rows are keyed by id,
    so one image shared by several documents is captioned once. Idempotent: only
    rows still lacking a caption are touched, so a resumed run fills gaps.

    Analysis (OCR + VLM) runs concurrently across ``caption_workers`` threads —
    the network-bound work the GIL releases — while embedding and the DB write
    stay on this thread, the Writer's sole owner. At ``caption_workers <= 1`` the
    work runs inline: the path ``--timings`` forces, for a clean sequential
    baseline with per-document caption seconds.

    Mutates nothing on the units except their ``stages`` (under timings); each
    document's remaining incompleteness is recomputed from the DB in phase 3.
    """
    # Unique uncaptioned rows across every unit; the first unit referencing a row
    # owns its timing. A row shared across documents appears once.
    pending: dict[int, PendingImage] = {}
    owner: dict[int, _DocUnit] = {}
    for unit in units:
        for pi in writer.uncaptioned_images(unit.document_id):
            if pi.image_id not in pending:
                pending[pi.image_id] = pi
                owner[pi.image_id] = unit
    if not pending:
        return
    if not vision_enabled:
        console.warn(
            f"Skipping {len(pending)} uncaptioned image(s) — no vision provider "
            f"configured."
        )
        return

    total = len(pending)
    done = 0
    if on_progress is not None:
        on_progress(0, total)

    def _advance() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    # Capped rows (failed MAX_INGEST_ATTEMPTS× already) are skipped, not retried.
    to_caption: dict[int, PendingImage] = {}
    for image_id, pi in pending.items():
        if writer.is_capped(pi.file_hash, "caption"):
            console.warn(
                f"{owner[image_id].file_name}: skipping image — failed "
                f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
            )
            _advance()
        else:
            to_caption[image_id] = pi

    def _persist(image_id: int, analysis: image_pipeline.ImageAnalysis) -> None:
        pi = to_caption[image_id]
        writer.persist_caption(_caption_from_analysis(pi, analysis, vision_model))
        writer.clear_failure(pi.file_hash, "caption")

    def _fail(image_id: int, exc: Exception) -> None:
        unit = owner[image_id]
        console.warn(f"{unit.file_name}: image analysis failed: {exc}")
        writer.record_failure(
            to_caption[image_id].file_hash, unit.file_name, "caption", exc
        )

    if caption_workers <= 1:
        # Inline: one image at a time on this thread. Same analyze/persist calls
        # as the pool, but with clean per-document caption seconds for --timings.
        for image_id, pi in to_caption.items():
            t0 = time.perf_counter() if timings else None
            try:
                _persist(image_id, _analyze_image(
                    pi, vision_provider=vision_provider, vision_model=vision_model,
                ))
            except Exception as e:
                _fail(image_id, e)
            unit = owner[image_id]
            if t0 is not None and unit.stages is not None:
                unit.stages["caption"] = (
                    unit.stages.get("caption", 0.0) + (time.perf_counter() - t0)
                )
            _advance()
        return

    with ThreadPoolExecutor(max_workers=caption_workers) as pool:
        futures = {
            pool.submit(
                _analyze_image, pi,
                vision_provider=vision_provider, vision_model=vision_model,
            ): image_id
            for image_id, pi in to_caption.items()
        }
        for fut in as_completed(futures):
            image_id = futures[fut]
            try:
                _persist(image_id, fut.result())
            except Exception as e:
                _fail(image_id, e)
            _advance()


def _summarize_all(
    writer: Writer,
    pending: list[PendingSummary],
    *,
    llm_provider: Provider | None,
    llm_model: str | None,
    temperature: float,
    max_summarize_tokens: int,
    summarize_workers: int,
    timings: bool,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[int, dict[int, tuple[str, float]]]:
    """Phase 3: summarize every document still owing one, concurrently (#188).

    Mirrors the caption stage (#166): the network-bound ``summarize()`` LLM call
    fans out across ``summarize_workers`` threads while the Writer's persist stays
    on this thread, its connection's sole owner. Capped documents are surfaced as
    still-incomplete rather than retried; a single failure never tears down the
    pass. The ``pending`` work-list is :meth:`Writer.documents_needing_summary`,
    which already excludes summarized and zero-chunk docs (issue #80), so the only
    main-thread guard left is the cap check.

    One deliberate difference from captioning: a caption's payload is an on-disk
    image the worker reads itself, but a summary's payload is assembled from the
    DB by ``writer.summary_input`` — a Writer-owned read that must stay on this
    thread. So inputs are fetched here and only ``summarize_workers`` are kept in
    flight, holding the old sequential pass's flat memory footprint instead of
    materializing every document's text at once.

    Returns ``(incomplete_count, {document_id: (file_name, seconds)})`` — the
    second map is populated only under ``timings`` (which forces one worker, so
    the per-document seconds stay a clean sequential baseline).
    """
    total = len(pending)
    done = 0
    incomplete = 0
    times: dict[int, tuple[str, float]] = {}
    if on_progress is not None:
        on_progress(0, total)

    def _advance() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    # Capped docs (failed MAX_INGEST_ATTEMPTS× already) are surfaced, not retried.
    work: list[PendingSummary] = []
    for ps in pending:
        if writer.is_capped(ps.file_hash, "summary"):
            console.warn(
                f"{ps.file_name}: skipping summary — failed "
                f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
            )
            incomplete += 1
            _advance()
        else:
            work.append(ps)

    def _persist(ps: PendingSummary, result) -> None:
        writer.persist_summary(
            ps.document_id, result, _summary_chunks(result.text)
        )
        writer.clear_failure(ps.file_hash, "summary")

    def _fail(ps: PendingSummary, exc: Exception) -> None:
        nonlocal incomplete
        console.warn(f"{ps.file_name}: summary failed: {exc}")
        writer.record_failure(ps.file_hash, ps.file_name, "summary", exc)
        incomplete += 1

    def _summarize(text: str):
        return summarize(
            text, provider=llm_provider, model=llm_model,
            temperature=temperature, max_summarize_tokens=max_summarize_tokens,
        )

    if summarize_workers <= 1:
        # Inline: one document at a time on this thread. Same summarize/persist
        # calls as the pool, but with clean per-document seconds for --timings.
        for ps in work:
            t0 = time.perf_counter() if timings else None
            try:
                _persist(ps, _summarize(writer.summary_input(ps.document_id)))
            except Exception as e:
                _fail(ps, e)
            if t0 is not None:
                times[ps.document_id] = (ps.file_name, time.perf_counter() - t0)
            _advance()
        return incomplete, times

    # Pooled: keep at most ``summarize_workers`` inputs in flight (see docstring) —
    # fetch each document's text on this thread, then top up one as each completes.
    it = iter(work)
    with ThreadPoolExecutor(max_workers=summarize_workers) as pool:
        in_flight = {
            pool.submit(_summarize, writer.summary_input(ps.document_id)): ps
            for ps in itertools.islice(it, summarize_workers)
        }
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                ps = in_flight.pop(fut)
                try:
                    _persist(ps, fut.result())
                except Exception as e:
                    _fail(ps, e)
                _advance()
                nxt = next(it, None)
                if nxt is not None:
                    in_flight[
                        pool.submit(_summarize, writer.summary_input(nxt.document_id))
                    ] = nxt
    return incomplete, times


# -------------------- parse pool: requests, config, worker fn --------------------


@dataclass(frozen=True)
class ParseConfig:
    """Run-wide parse settings — identical for every document, shipped to each
    worker once. Picklable scalars only (it crosses the spawn boundary)."""
    pdf_converter: str
    html_converter: str
    sparse_text_threshold: int
    ocr_min_confidence: int
    vision_enabled: bool
    vision_max_dimension: int
    vision_min_dimension: int
    archive_root: Path
    timings: bool = False


@dataclass
class ParseRequest:
    """One file to parse — the per-document unit the pool schedules."""
    path: Path
    ext: str
    file_hash: str
    file_name: str


@dataclass
class ParseOutcome:
    """A parse result (or the failure that stood in for it), returned to the
    main-process drain. ``parsed`` is None exactly when ``error`` is set."""
    request: ParseRequest
    parsed: ParsedDocument | None = None
    parse_stages: dict[str, float] | None = None  # canonical stage→seconds (--timings)
    error: str | None = None
    offline: bool = False  # the error was an offline-mode block (hint the user)


@dataclass
class _ResumeItem:
    """A document parsed by an earlier run that still has missing units."""
    document_id: int
    file_name: str
    file_hash: str


def _parse_request(request: ParseRequest, config: ParseConfig) -> ParseOutcome:
    """Archive + parse one file into a ParsedDocument — the pool's ``parse_fn``.

    DB-free, so it runs anywhere the pool puts it (a spawn worker, or inline at
    ``max_workers <= 1``). It never raises: a parse failure comes back as an
    ``error`` outcome so one bad file can't tear down the result stream. With
    ``--timings`` it captures its own parse/embed wall-clock (the timer is built
    here, in whatever process runs the parse) and returns it as data.
    """
    timer = timing.StageTimer() if config.timings else None
    try:
        parsed = _parse_document(
            _archive(request.path, config.archive_root, request.file_hash, request.ext),
            request.ext, file_hash=request.file_hash, file_name=request.file_name,
            pdf_converter=config.pdf_converter, html_converter=config.html_converter,
            sparse_text_threshold=config.sparse_text_threshold,
            ocr_min_confidence=config.ocr_min_confidence,
            vision_enabled=config.vision_enabled,
            vision_max_dimension=config.vision_max_dimension,
            vision_min_dimension=config.vision_min_dimension,
            archive_root=config.archive_root,
            on_stage=timer.mark if timer is not None else None,
        )
    except Exception as e:
        from bartleby.lib.quiet import offline_blocked
        return ParseOutcome(request=request, error=str(e), offline=offline_blocked(e))

    if timer is not None:
        timer.finish()
    return ParseOutcome(
        request=request, parsed=parsed,
        parse_stages=dict(timer.totals) if timer is not None else None,
    )


def _warm_worker(config: ParseConfig) -> None:
    """Pool-initializer hook: load the models this run needs, once, up front, so
    a worker's first document doesn't pay the load. Embeddings always; docling's
    layout/table models only when a docling converter is active."""
    from bartleby.ingest import embed

    embed.prewarm()
    if "docling" in (config.pdf_converter, config.html_converter):
        from bartleby.ingest import docling as docling_pipeline

        docling_pipeline.prewarm(config.vision_enabled)


def _classify(
    writer: Writer,
    sources: list[tuple[Path, str]],
    *,
    vision_enabled: bool,
) -> tuple[list[ParseRequest], list[_ResumeItem], list[str]]:
    """Bucket each source by what the parse/caption drain still needs, all from
    DB state: already parsed+captioned (skip), parsed-but-uncaptioned (resume —
    no parse), or never parsed (hand to the pool). Summaries are not considered
    here — they're settled by their own pass over whatever the DB still lacks.
    Hashing + the resume lookups run here on the main process; only the parse
    bucket crosses to workers."""
    to_parse: list[ParseRequest] = []
    to_resume: list[_ResumeItem] = []
    skipped: list[str] = []
    for path, ext in sources:
        file_hash = _hash_file(path)
        document_id = writer.document_id_for(file_hash)
        if document_id is not None:
            if _is_complete(writer, document_id):
                skipped.append(path.name)
            else:
                to_resume.append(_ResumeItem(document_id, path.name, file_hash))
            continue
        if ext in IMAGE_EXTENSIONS and not vision_enabled:
            console.warn(
                f"{path.name}: skipping image (no vision provider configured)."
            )
            continue
        to_parse.append(
            ParseRequest(path=path, ext=ext, file_hash=file_hash, file_name=path.name)
        )
    return to_parse, to_resume, skipped


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


def _resolve_max_workers(config: dict, *, timings: bool) -> int:
    """Resolve how many parse workers to run.

    An explicit ``max_workers`` in config wins (warning when it exceeds what the
    machine can safely hold); otherwise auto-pick ``min(cpu_count, free_ram_gb //
    PER_WORKER_GB)``, floored at 1, so a CPU-rich but RAM-poor box doesn't launch
    more workers than memory can hold and OOM. ``--timings`` forces 1: the
    per-stage breakdown it produces is a sequential baseline, meaningless once
    documents parse concurrently and overlap.
    """
    if timings:
        if int(config.get("max_workers") or 1) > 1:
            console.warn("--timings runs sequentially (max_workers=1) for a clean baseline.")
        return 1

    import psutil

    cores = os.cpu_count() or 1
    free_gb = psutil.virtual_memory().available / 1024**3
    auto = max(1, min(cores, int(free_gb // PER_WORKER_GB)))

    configured = config.get("max_workers")
    if configured is None:
        return auto
    n = max(1, int(configured))
    if n > auto:
        console.warn(
            f"max_workers={n} exceeds what this machine can comfortably run "
            f"({cores} cores, ~{free_gb:.0f}GB free → {auto} by the RAM cap); "
            f"proceeding, but watch for memory pressure."
        )
    return n


def _resolve_caption_workers(config: dict, *, timings: bool) -> int:
    """Resolve how many images caption concurrently in the post-parse stage.

    Unlike parse workers (RAM-bound, auto-sized), captioning is network/IO-bound,
    so this is a plain configured count defaulting to ``DEFAULT_CAPTION_WORKERS``.
    ``--timings`` forces 1: the per-stage breakdown is a sequential baseline,
    meaningless once captions overlap (same rationale as ``max_workers``).
    """
    configured = int(config.get("caption_workers") or DEFAULT_CAPTION_WORKERS)
    if timings:
        if configured > 1:
            console.warn(
                "--timings captions sequentially (caption_workers=1) for a "
                "clean baseline."
            )
        return 1
    return max(1, configured)


def _resolve_summarize_workers(config: dict, *, timings: bool) -> int:
    """Resolve how many documents summarize concurrently in the post-parse pass.

    Like captioning (and unlike RAM-bound parse workers), summarization is
    network/IO-bound, so this is a plain configured count defaulting to
    ``DEFAULT_SUMMARIZE_WORKERS``. ``--timings`` forces 1 for a clean per-document
    baseline, meaningless once summaries overlap (same rationale as the others).
    """
    configured = int(config.get("summarize_workers") or DEFAULT_SUMMARIZE_WORKERS)
    if timings:
        if configured > 1:
            console.warn(
                "--timings summarizes sequentially (summarize_workers=1) for a "
                "clean baseline."
            )
        return 1
    return max(1, configured)


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

    parse_config = ParseConfig(
        pdf_converter=pdf_converter_name,
        html_converter=html_converter_name,
        sparse_text_threshold=int(
            config.get("sparse_text_threshold", DEFAULT_SPARSE_TEXT_THRESHOLD)
        ),
        ocr_min_confidence=int(
            config.get("ocr_min_confidence", DEFAULT_OCR_MIN_CONFIDENCE)
        ),
        vision_enabled=vision_provider is not None and bool(vision_model),
        vision_max_dimension=int(
            config.get("vision_max_dimension", DEFAULT_VISION_MAX_DIMENSION)
        ),
        vision_min_dimension=int(
            config.get("vision_min_dimension", DEFAULT_VISION_MIN_DIMENSION)
        ),
        archive_root=archive_root,
        timings=timings,
    )
    temperature = float(config.get("temperature", 0))
    max_summarize_tokens = int(config.get("max_summarize_tokens", 50_000))
    summaries_enabled = llm_provider is not None and bool(llm_model)

    conn = open_db(project_name)
    writer = Writer(conn)
    try:
        # Record this invocation's resolved config (secrets stripped) so resume
        # is auditable per unit; warn — never block — on any field that drifted
        # since the last ingest, read *before* this run's row is inserted.
        config_snapshot = redact_config({
            **config,
            "pdf_converter": pdf_converter_name,
            "html_converter": html_converter_name,
            "provider": provider or config.get("provider"),
            "model": model or config.get("model"),
        })
        for line in config_drift(writer.latest_config(), config_snapshot):
            console.warn(f"Config drift since last ingest — {line}")
        writer.begin_run(config_snapshot)

        # Classify up front (hash + resume lookups, on this process): only files
        # that have never been parsed cross to the workers. Parsed-but-uncaptioned
        # docs resume on the main process — they need no parse, just the missing
        # captions. Summaries are settled afterwards by their own pass.
        to_parse, to_resume, skipped = _classify(
            writer, sources,
            vision_enabled=parse_config.vision_enabled,
        )
        SKIP_DISPLAY_LIMIT = 3
        for name in skipped[:SKIP_DISPLAY_LIMIT]:
            console.info(f"Skipping {name} (already ingested)")
        if len(skipped) > SKIP_DISPLAY_LIMIT:
            console.info(
                f"… and {len(skipped) - SKIP_DISPLAY_LIMIT} more file(s) "
                f"skipped as already ingested"
            )

        max_workers = _resolve_max_workers(config, timings=timings) if to_parse else 1
        caption_workers = _resolve_caption_workers(config, timings=timings)
        summarize_workers = _resolve_summarize_workers(config, timings=timings)
        if len(to_parse) > 1 and max_workers > 1:
            console.info(
                f"Parsing {len(to_parse)} file(s) across {max_workers} workers"
            )

        # --timings: one record per document, keyed by document_id and built
        # across both passes — parse/embed/caption in the drain, summarize in
        # the summarize pass — then emitted once at the end. Insertion order is
        # drain order. (Only populated under --timings, which forces 1 worker.)
        doc_timings: dict[int, tuple[str, timing.DocTiming]] = {}
        run_start = time.perf_counter() if timings else None

        # Three sequential phases behind the single Writer: parse (pooled or
        # inline) → caption every image at once (#166: a concurrent stage decoupled
        # from parse) → summarize. The barriers between them keep the Writer the
        # connection's sole owner on one thread; only parse and caption fan out,
        # each to its own kind of worker (parse processes, caption threads).
        # Share the console module's Rich Console so messages printed via
        # console.info/error during the bar's lifetime insert above the Live
        # display rather than colliding with it (and keep stdout-redirect captures
        # around Docling/RapidOCR from breaking the bar).
        with Progress(
            TextColumn("[bold]{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console.get_console(),
        ) as bar:
            task = bar.add_task("phase", total=len(to_parse), label="Parsing")

            # ---- Phase 1: parse + persist ------------------------------------
            # Resumed docs were parsed by an earlier run — no parse, just the
            # missing captions the caption phase fills; summaries are settled by
            # their own pass over the DB (#167). Under --timings each unit's record
            # starts from parse and gains caption/summarize downstream.
            incomplete_count = 0
            units: list[_DocUnit] = [
                _DocUnit(
                    item.document_id, item.file_name, item.file_hash,
                    stages={} if timings else None,
                )
                for item in to_resume
            ]
            for outcome in pool.parse_stream(
                to_parse,
                parse_fn=_parse_request,
                config=parse_config,
                max_workers=max_workers,
                warmup=_warm_worker,
                verbose=verbose,
                required_models=_required_hf_models(
                    pdf_converter_name, html_converter_name
                ),
            ):
                req = outcome.request
                if outcome.error is not None:
                    writer.record_failure(
                        req.file_hash, req.file_name, "parse", outcome.error
                    )
                    message = f"Failed: {req.file_name}: {outcome.error}"
                    if outcome.offline:
                        from bartleby.lib.quiet import OFFLINE_HINT
                        message += f"\n  {OFFLINE_HINT}"
                    console.error(message)
                    bar.advance(task)
                    continue

                writer.clear_failure(req.file_hash, "parse")
                persist_t = time.perf_counter()
                document_id = writer.persist_parse(outcome.parsed)
                stages = None
                if timings and outcome.parse_stages is not None:
                    # Chunk INSERTs fold into `embed`, per #162 — they land here,
                    # just after the worker's embed step, under the same bucket.
                    stages = dict(outcome.parse_stages)
                    stages["embed"] = (
                        stages.get("embed", 0.0) + (time.perf_counter() - persist_t)
                    )
                units.append(_DocUnit(
                    document_id, req.file_name, req.file_hash,
                    page_count=outcome.parsed.page_count, stages=stages,
                ))
                bar.advance(task)

            # ---- Phase 2: caption every uncaptioned image, concurrently ------
            def _caption_progress(done: int, total: int) -> None:
                if done == 0:
                    bar.reset(task, total=total, label="Captioning images")
                else:
                    bar.update(task, completed=done)

            _caption_all(
                writer, units,
                vision_provider=vision_provider, vision_model=vision_model,
                vision_enabled=parse_config.vision_enabled,
                caption_workers=caption_workers, timings=timings,
                on_progress=_caption_progress,
            )
            # A document is caption-incomplete if any image is still uncaptioned
            # (failed / capped / no provider — recomputed from the DB so a shared
            # image counts against every document holding it). Seed each unit's
            # timing record now that parse + caption are final; the summarize pass
            # adds its stage by document_id.
            for unit in units:
                if writer.uncaptioned_images(unit.document_id):
                    incomplete_count += 1
                if timings and unit.stages is not None:
                    doc_timings[unit.document_id] = (
                        unit.file_name,
                        timing.DocTiming(
                            page_count=unit.page_count, stages=unit.stages,
                        ),
                    )

            # ---- Phase 3: summarize as its own concurrent pass (#167, #188) --
            # Over whatever the DB still lacks — keeps the slow LLM summary off the
            # parse/caption critical path; resume falls out for free, since the
            # work-list is simply what's missing. Like captioning, it fans out
            # across its own workers (#188) with the Writer's persist on this
            # thread.
            if summaries_enabled:
                pending = writer.documents_needing_summary()
                if pending:
                    stask = bar.add_task(
                        "summaries", total=len(pending), label="Summarizing",
                    )

                    def _summary_progress(done: int, total: int) -> None:
                        bar.update(stask, completed=done, total=total)

                    owed, summarize_times = _summarize_all(
                        writer, pending,
                        llm_provider=llm_provider, llm_model=llm_model,
                        temperature=temperature,
                        max_summarize_tokens=max_summarize_tokens,
                        summarize_workers=summarize_workers, timings=timings,
                        on_progress=_summary_progress,
                    )
                    incomplete_count += owed
                    if timings:
                        for doc_id, (fname, secs) in summarize_times.items():
                            name, rec = doc_timings.get(
                                doc_id,
                                (fname, timing.DocTiming(page_count=None)),
                            )
                            rec.stages["summarize"] = (
                                rec.stages.get("summarize", 0.0) + secs
                            )
                            doc_timings[doc_id] = (name, rec)

            if incomplete_count:
                console.warn(
                    f"{incomplete_count} file(s) ingested but incomplete — "
                    f"some unit(s) still missing (see below)."
                )

        _report_failures(writer)
    finally:
        writer.finish_run()
        conn.close()

    if timings:
        # One coherent line per document (parse + caption + summarize merged),
        # emitted now that both passes have run. Human breakdown to stderr;
        # machine summary to stdout so a benchmark run is captured with
        # `bartleby scribe --files <sample> --timings > bench.json` while the bar
        # and status text stay on stderr.
        records: list[timing.DocTiming] = []
        for name, rec in doc_timings.values():
            console.info(
                timing.render_doc_line(name, sum(rec.stages.values()), rec.stages)
            )
            records.append(rec)
        agg = timing.aggregate(records, time.perf_counter() - run_start)
        console.big("Timing summary")
        for line in timing.render_summary(agg):
            console.info(line)
        print(json.dumps(agg), flush=True)

    console.complete("Done.")
