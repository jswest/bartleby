"""DB-free document parsing for the scribe ingest pipeline.

Pulled out of ``commands/scribe.py`` (#306): everything here converts an
archived file into a :class:`~bartleby.ingest.writer.ParsedDocument` without ever
touching the database — the per-type converters, the image-routing helper, the
chunk/token plumbing they share, and the pool scaffolding (``ParseConfig`` /
``ParseRequest`` / ``ParseOutcome`` plus the ``parse_fn``/``warmup`` the pool
schedules). Being DB-free is what lets ``_parse_request`` run in a spawn worker
or inline at ``max_workers <= 1``; the single Writer drains the results back in
the main process.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from bartleby.db.chunks import ChunkInput
from bartleby.ingest import edgar as edgar_pipeline
from bartleby.ingest import embed
from bartleby.ingest import images as image_pipeline
from bartleby.ingest import pdfplumber as pdfplumber_pipeline
from bartleby.ingest import sec2md as sec2md_pipeline
from bartleby.ingest.chunk import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    ChunkRow,
    convert_and_chunk,
)
from bartleby.ingest.summarize import count_tokens
from bartleby.ingest.text import chunk_text
from bartleby.ingest.writer import ParsedDocument, ParsedImage, ParsedSection
from bartleby.lib import timing


HTML_EXTENSIONS = {".html", ".htm"}

def _archive(src: Path, archive_root: Path, file_hash: str, ext: str) -> Path:
    dest_dir = archive_root / file_hash
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{file_hash}{ext}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest

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
    config: ParseConfig,
    *,
    on_warn: Callable[[str], None] | None = None,
) -> list[ParsedImage]:
    """Scale + archive each route into a ParsedImage (no VLM, no DB).

    Sub-minimum images are dropped here, exactly as before: VLM image processors
    crash on sub-patch-size edges and such slivers carry no describable content.
    With no vision provider there's nothing to caption, so routes produce no
    rows (a warning, then skipped). Notices go to ``on_warn`` (routed to the
    parent), never the console — this runs in a spawn worker with no Live.
    """
    if not routes:
        return []
    if not config.vision_enabled:
        if on_warn is not None:
            on_warn(
                f"Skipping {len(routes)} image(s) — no vision provider configured."
            )
        return []
    parsed: list[ParsedImage] = []
    for route in routes:
        prepared = image_pipeline.prepare_image(
            route.bytes_, max_dimension=config.vision_max_dimension,
        )
        if image_pipeline.is_below_vlm_minimum(
            prepared, min_dimension=config.vision_min_dimension
        ):
            page_str = f" (page {route.page_number})" if route.page_number else ""
            if on_warn is not None:
                on_warn(
                    f"Skipping image{page_str} — {prepared.width}x{prepared.height}px "
                    f"is below the {config.vision_min_dimension}px vision minimum."
                )
            continue
        archived = image_pipeline.archive_image(prepared, config.archive_root)
        parsed.append(ParsedImage(
            hash=prepared.hash,
            archive_path=archived,
            width=prepared.width,
            height=prepared.height,
            page_number=route.page_number,
            image_index_on_page=route.image_index_on_page,
        ))
    return parsed

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
        embeddings = embed.embed_texts([row.text for row in result.chunks])
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

def _section_file_hash(file_bytes: bytes, anchor_id: str) -> str:
    """The derived hash for a section row: ``sha256(file_bytes + anchor_id)``.

    Keeps ``documents.file_hash`` UNIQUE and re-ingest-stable, and cannot collide
    with the container's plain ``sha256(file_bytes)`` or a standalone copy of the
    section file's own byte-hash (#254)."""
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(anchor_id.encode("utf-8"))
    return h.hexdigest()

def _build_sec2md_chunks(result, *, fallback_heading: str | None = None) -> list[ChunkInput]:
    """Embed a ``Sec2mdResult``'s chunks into ChunkInputs (empty when none)."""
    if not result.chunks:
        return []
    rows = [_sec2md_chunk_to_row(c, fallback_heading=fallback_heading)
            for c in result.chunks]
    embeddings = embed.embed_texts([r.text for r in rows])
    return _build_chunk_inputs(rows, embeddings)

def _parse_html_sec2md(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """iXBRL EDGAR filing via sec2md.

    When the filing carries a usable table of contents, it is split at its
    internal anchors into a zero-chunk container document plus N section
    documents (#254); otherwise it ingests whole, exactly as before. A
    non-anchored filing returns one ordinary ParsedDocument.
    """
    if on_stage is not None:
        on_stage("extracting")
    file_bytes = archived.read_bytes()
    sections = sec2md_pipeline.convert_sections_bytes(file_bytes)
    if sections:
        return _parse_html_sec2md_split(
            archived, sections, file_bytes=file_bytes,
            file_hash=file_hash, file_name=file_name, on_stage=on_stage,
        )
    result = sec2md_pipeline.convert_bytes(file_bytes)
    if on_stage is not None and result.chunks:
        on_stage("embedding")
    chunks = _build_sec2md_chunks(result)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=result.page_count, token_count=_token_count(result.full_text),
        document_chunks=chunks, images=[],
    )

def _parse_html_sec2md_split(
    archived: Path,
    sections: list,
    *,
    file_bytes: bytes,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """Build the container + section ParsedDocument for a TOC-anchored filing.

    The container holds the original ``file_hash`` and **no chunks of its own**;
    each section gets a derived hash and its own embedded chunks. The whole thing
    is one atomic write unit — the Writer persists the container last."""
    if on_stage is not None:
        on_stage("embedding")
    parsed_sections: list[ParsedSection] = []
    total_tokens = 0
    for sec in sections:
        chunks = _build_sec2md_chunks(sec.result, fallback_heading=sec.title)
        token_count = _token_count(sec.result.full_text)
        total_tokens += token_count
        parsed_sections.append(ParsedSection(
            file_hash=_section_file_hash(file_bytes, sec.anchor_id),
            anchor_id=sec.anchor_id,
            section_title=sec.title,
            section_order=sec.order,
            token_count=token_count,
            document_chunks=chunks,
        ))
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=None, token_count=total_tokens,
        document_chunks=[], images=[], sections=parsed_sections,
    )

def _parse_edgar_submission(
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
    on_warn: Callable[[str], None] | None = None,
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
            if on_warn is not None:
                on_warn(f"{file_name}: skipping inner document {label}.")
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
        embeddings = embed.embed_texts([r.text for r in rows])
        chunks = _build_chunk_inputs(rows, embeddings)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=None, token_count=_token_count(full_text),
        document_chunks=chunks, images=[],
    )

def _parse_pdf_pdfplumber(
    archived: Path,
    config: ParseConfig,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
    on_warn: Callable[[str], None] | None = None,
) -> ParsedDocument:
    if on_stage is not None:
        on_stage("extracting")
    result = pdfplumber_pipeline.convert(
        archived,
        sparse_text_threshold=config.sparse_text_threshold,
        ocr_min_confidence=config.ocr_min_confidence,
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
        embeddings = embed.embed_texts([r.text for r in doc_rows])
        chunks = _build_chunk_inputs(doc_rows, embeddings)

    images = _parse_image_routes(image_routes, config, on_warn=on_warn)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=result.page_count, token_count=_token_count(result.full_text),
        document_chunks=chunks, images=images,
    )

def _parse_pdf_docling(
    archived: Path,
    config: ParseConfig,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
    on_warn: Callable[[str], None] | None = None,
) -> ParsedDocument:
    from bartleby.ingest import docling as docling_pipeline

    if on_stage is not None:
        on_stage("extracting")
    docling_result = docling_pipeline.convert(
        archived, extract_images=config.vision_enabled,
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
        embeddings = embed.embed_texts([r.text for r in rows])
        chunks = _build_chunk_inputs(rows, embeddings)

    # Embedded images come out of the same docling pass (no second parse).
    image_routes: list[_ImageRoute] = []
    if config.vision_enabled and docling_result.images:
        image_routes = [
            _ImageRoute(
                bytes_=img.png_bytes,
                page_number=img.page_number,
                image_index_on_page=img.image_index_on_page,
            )
            for img in docling_result.images
        ]
    images = _parse_image_routes(image_routes, config, on_warn=on_warn)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=docling_result.page_count,
        token_count=_token_count(docling_result.full_text),
        document_chunks=chunks, images=images,
    )

def _parse_image_file(
    archived: Path,
    config: ParseConfig,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
    on_warn: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """A standalone image: one route, no document chunks. The summarizer (if
    enabled) builds its input from the image chunk once captioned. With vision
    off the route produces no rows (warned + skipped in ``_parse_image_routes``),
    so the document lands empty rather than stranding an uncaptionable row."""
    if on_stage is not None:
        on_stage("extracting")
    route = _ImageRoute(
        bytes_=archived.read_bytes(), page_number=None, image_index_on_page=0,
    )
    images = _parse_image_routes([route], config, on_warn=on_warn)
    return ParsedDocument(
        file_hash=file_hash, file_name=file_name, archive_path=archived,
        page_count=None, token_count=0, document_chunks=[], images=images,
    )

def _parse_document(
    archived: Path,
    ext: str,
    config: ParseConfig,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
    on_warn: Callable[[str], None] | None = None,
) -> ParsedDocument:
    """Route an archived file to its converter and return a parsed result."""
    if ext in IMAGE_EXTENSIONS:
        return _parse_image_file(
            archived, config, file_hash=file_hash, file_name=file_name,
            on_stage=on_stage, on_warn=on_warn,
        )
    if ext in PDF_EXTENSIONS:
        # Reject HTML error pages saved with a `.pdf` extension before either
        # backend dies deep in pdfminer with a cryptic "No /Root object!" — and
        # *without* rerouting them into the HTML pipeline (they carry no
        # document content; ingesting a portal error page pollutes the corpus).
        pdfplumber_pipeline.reject_if_html(archived)
        if config.pdf_converter == "docling":
            return _parse_pdf_docling(
                archived, config, file_hash=file_hash, file_name=file_name,
                on_stage=on_stage, on_warn=on_warn,
            )
        return _parse_pdf_pdfplumber(
            archived, config, file_hash=file_hash, file_name=file_name,
            on_stage=on_stage, on_warn=on_warn,
        )
    if edgar_pipeline.detect(archived):
        # EDGAR full-submission SGML envelope — detected by content, so a `.txt`
        # wrapper is caught here instead of falling to the character chunker.
        return _parse_edgar_submission(
            archived, file_hash=file_hash, file_name=file_name,
            on_stage=on_stage, on_warn=on_warn,
        )
    if (
        ext in HTML_EXTENSIONS
        and config.html_converter == "sec2md"
        and sec2md_pipeline.is_ixbrl(archived)
    ):
        return _parse_html_sec2md(
            archived, file_hash=file_hash, file_name=file_name, on_stage=on_stage,
        )
    return _parse_text_document(
        archived, file_hash=file_hash, file_name=file_name, on_stage=on_stage,
    )

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
    # User-facing notices collected during parse. A spawn worker has no access
    # to the parent's Rich Live console, so it never prints — it routes them
    # here and the main-process drain emits them (see _parse_request).
    warnings: list[str] = field(default_factory=list)

def _parse_request(
    request: ParseRequest,
    config: ParseConfig,
    report: Callable[[str], None],
) -> ParseOutcome:
    """Archive + parse one file into a ParsedDocument — the pool's ``parse_fn``.

    DB-free, so it runs anywhere the pool puts it (a spawn worker, or inline at
    ``max_workers <= 1``). It never raises: a parse failure comes back as an
    ``error`` outcome so one bad file can't tear down the result stream. With
    ``--timings`` it captures its own parse/embed wall-clock (the timer is built
    here, in whatever process runs the parse) and returns it as data.

    ``report(stage)`` is the pool's progress hook: it announces the file's
    current stage so the main process can render this worker's lane. The single
    ``on_stage`` the parsers already emit drives both the timer and ``report`` —
    one callback, two consumers.

    User-facing notices (skipped images, skipped inner documents) accumulate via
    ``on_warn`` and ride back on the outcome: a spawn worker has no Live console,
    so the main-process drain is the only place that can print them.
    """
    timer = timing.StageTimer() if config.timings else None
    report("preparing")          # the lane shows the file the moment we pick it up

    def on_stage(label: str) -> None:
        if timer is not None:
            timer.mark(label)
        report(label)

    warnings: list[str] = []
    try:
        parsed = _parse_document(
            _archive(request.path, config.archive_root, request.file_hash, request.ext),
            request.ext, config, file_hash=request.file_hash,
            file_name=request.file_name,
            on_stage=on_stage, on_warn=warnings.append,
        )
    except Exception as e:
        from bartleby.lib.quiet import offline_blocked
        return ParseOutcome(
            request=request, error=str(e), offline=offline_blocked(e),
            warnings=warnings,
        )

    if timer is not None:
        timer.finish()
    return ParseOutcome(
        request=request, parsed=parsed,
        parse_stages=dict(timer.totals) if timer is not None else None,
        warnings=warnings,
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
