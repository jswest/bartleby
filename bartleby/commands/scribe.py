"""`bartleby scribe` — sequential ingest pipeline.

Hash → archive → convert → embed → (optionally) summarize → write. All chunk
writes go through ``bartleby.db.chunks`` typed helpers; image rows additionally
get a join entry in ``document_images``.

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
import shutil
import sys
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

from bartleby.config import ensure_provider_env, load_config
from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_document_chunks,
    insert_image_chunks,
    insert_summary_chunks,
)
from bartleby.db.connection import open_db, resolve_project_name
from bartleby.ingest import edgar as edgar_pipeline
from bartleby.ingest import images as image_pipeline
from bartleby.ingest import ocr as ocr_module
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
from bartleby.ingest.summarize import SummaryResult, count_tokens, summarize
from bartleby.ingest.text import chunk_text
from bartleby.lib import console
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


def _document_already_ingested(conn, file_hash: str) -> int | None:
    """Return document_id iff this hash has a *complete* ingest.

    Completeness = the document has at least one text chunk OR at least one
    image attached. A bare ``documents`` row with neither is a stranded
    partial from an interrupted run (the row is committed in autocommit
    before chunks/images land); treating it as "ingested" would mean
    skipping a broken document forever. Returning None here lets the
    caller clean up the stranded row and re-ingest from scratch.
    """
    row = conn.cursor().execute(
        "SELECT d.document_id "
        "FROM documents d "
        "WHERE d.file_hash = ? AND ("
        "  EXISTS (SELECT 1 FROM chunks "
        "          WHERE source_kind = 'document' AND source_id = d.document_id)"
        "  OR EXISTS (SELECT 1 FROM document_images "
        "             WHERE document_id = d.document_id)"
        ")",
        (file_hash,),
    ).fetchone()
    return row[0] if row else None


def _cleanup_partial_document(conn, document_id: int) -> None:
    """Drop a stranded document row and its dependent state.

    FK CASCADE handles summaries and document_images; the polymorphic
    chunks (kind=document, kind=summary) are cleared manually via
    ``delete_chunks_for`` so chunks_fts and chunks_vec stay in sync.
    Shared image rows survive — they're deduped on their own file_hash
    and may be reused by other documents or by the upcoming re-ingest.
    """
    cur = conn.cursor()
    summary_row = cur.execute(
        "SELECT summary_id FROM summaries WHERE document_id = ?", (document_id,)
    ).fetchone()

    delete_chunks_for(conn, "document", document_id)
    if summary_row is not None:
        delete_chunks_for(conn, "summary", summary_row[0])

    with conn:
        conn.cursor().execute(
            "DELETE FROM documents WHERE document_id = ?", (document_id,)
        )


def _insert_document(
    conn,
    *,
    file_hash: str,
    file_name: str,
    archive_path: Path,
    page_count: int | None,
    full_text: str,
) -> int:
    token_count = count_tokens(full_text) if full_text else 0
    conn.cursor().execute(
        "INSERT INTO documents "
        "(file_hash, file_name, file_path, page_count, token_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_hash, file_name, str(archive_path), page_count, token_count),
    )
    return conn.last_insert_rowid()


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


# -------------------- summary writing --------------------


def _write_summary(conn, document_id: int, summary: SummaryResult) -> int:
    cur = conn.cursor()
    prior = cur.execute(
        "SELECT summary_id FROM summaries WHERE document_id = ?", (document_id,)
    ).fetchone()
    if prior:
        delete_chunks_for(conn, "summary", prior[0])
        cur.execute("DELETE FROM summaries WHERE summary_id = ?", (prior[0],))

    cur.execute(
        "INSERT INTO summaries "
        "(document_id, title, description, text, model, authored_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (document_id, summary.title, summary.description,
         summary.text, summary.model, summary.authored_date),
    )
    summary_id = conn.last_insert_rowid()

    rows = chunk_markdown_string(summary.text)
    if rows:
        embeddings = embed_texts([r.text for r in rows])
        insert_summary_chunks(conn, summary_id, _build_chunk_inputs(rows, embeddings))
    return summary_id


def _build_summary_input(conn, document_id: int, fallback_text: str) -> str:
    """Interleave document and image chunks in source order for the summarizer.

    Either set may be empty: a text-only doc has no image chunks, an image-only
    doc (a standalone image, or an image-only PDF whose pages all rendered to
    the VLM) has no document chunks. We build from whatever chunks exist so the
    summarizer always sees real, indexed content — never the raw ``full_text``,
    which for image-only docs is trace garbage (page labels, form-feed
    separators) and makes the model confabulate (issue #80). ``fallback_text``
    is returned only when there are no chunks of either kind, which the
    chunk-count guard in ``_maybe_summarize`` already excludes.
    """
    cur = conn.cursor()
    img_rows = cur.execute(
        "SELECT di.page_number, c.chunk_index, c.text "
        "FROM chunks c "
        "JOIN document_images di ON di.image_id = c.source_id "
        "WHERE c.source_kind = 'image' AND di.document_id = ?",
        (document_id,),
    ).fetchall()
    doc_rows = cur.execute(
        "SELECT page_number, chunk_index, text FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ?",
        (document_id,),
    ).fetchall()
    if not img_rows and not doc_rows:
        return fallback_text

    KIND_DOC, KIND_IMG = 0, 1
    entries = (
        [(p, KIND_DOC, ci, t) for p, ci, t in doc_rows]
        + [(p, KIND_IMG, ci, t) for p, ci, t in img_rows]
    )
    entries.sort(key=lambda r: (r[0] if r[0] is not None else -1, r[1], r[2]))

    parts: list[str] = []
    for page_number, kind, _chunk_index, text in entries:
        if kind == KIND_IMG:
            label = (
                f"[Image on page {page_number}]"
                if page_number is not None else "[Image]"
            )
            parts.append(f"{label}\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _document_chunk_count(conn, document_id: int) -> int:
    """Count indexed chunks attributable to a document: its own document chunks
    plus any image chunks joined through ``document_images``.

    Summary chunks are excluded — at summarize time the summary doesn't exist
    yet, and it would never be an *input* to the summarizer anyway. This is the
    "is there anything real to summarize?" test (issue #80): a document with
    zero chunks has no extractable content, so the only text left to feed the
    model is the trace ``full_text``, which makes it confabulate.
    """
    row = conn.cursor().execute(
        "SELECT "
        "  (SELECT COUNT(*) FROM chunks "
        "   WHERE source_kind = 'document' AND source_id = ?) "
        "+ (SELECT COUNT(*) FROM chunks c "
        "     JOIN document_images di ON di.image_id = c.source_id "
        "   WHERE c.source_kind = 'image' AND di.document_id = ?)",
        (document_id, document_id),
    ).fetchone()
    return row[0]


def _maybe_summarize(
    conn,
    document_id: int,
    full_text: str,
    *,
    provider: Provider | None,
    model: str | None,
    temperature: float,
    max_summarize_tokens: int,
    on_stage: Callable[[str], None] | None = None,
) -> None:
    if provider is None or not model:
        return
    if _document_chunk_count(conn, document_id) == 0:
        # No indexed chunks → nothing real to summarize. Handing the model the
        # trace `full_text` (page labels, form-feed separators from an
        # image-only / sub-threshold doc) makes it confabulate a plausible but
        # entirely fabricated summary (issue #80). Skip, and surface it so the
        # user can spot an ingestion failure (e.g. an image-only PDF needing OCR).
        file_name = conn.cursor().execute(
            "SELECT file_name FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]
        console.warn(
            f"{file_name}: no summary — no extractable content (0 chunks). "
            f"The document may be image-only and need OCR/re-ingest."
        )
        return
    if on_stage is not None:
        on_stage("summarizing")
    summary_input = _build_summary_input(conn, document_id, full_text)
    result = summarize(
        summary_input,
        provider=provider,
        model=model,
        temperature=temperature,
        max_summarize_tokens=max_summarize_tokens,
    )
    _write_summary(conn, document_id, result)


# -------------------- image ingestion (the inner loop) --------------------


@dataclass
class _ImageRoute:
    """One image (embedded or page-render) ready for the pipeline."""
    bytes_: bytes
    page_number: int | None       # None for standalone files
    image_index_on_page: int      # 0 for page-renders, 1+ for embedded; 0 for standalone too
    # For sparse-page renders we already ran Tesseract at the page level — pass
    # that result down so the image pipeline can skip its own Tesseract pre-pass.
    prefetched_ocr: ocr_module.OcrResult | None = None


def _process_image(
    conn,
    document_id: int,
    route: _ImageRoute,
    *,
    archive_root: Path,
    vision_provider: Provider,
    vision_model: str,
    vision_max_dimension: int,
    vision_min_dimension: int,
) -> None:
    """Run the §7 image pipeline on one image and persist it.

    Dedupes on the image's source-byte hash: if we have already analyzed an
    identical image (across any document), reuse the ``images`` row and add a
    new ``document_images`` join. Chunks come along automatically since they
    point at ``image_id``.

    Images with an edge below ``vision_min_dimension`` are skipped entirely —
    no analysis, no archive, no rows — because VLMs crash on sub-patch-size
    images and such slivers carry no describable content anyway.
    """
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
        return
    cur = conn.cursor()

    existing = cur.execute(
        "SELECT image_id FROM images WHERE file_hash = ?", (prepared.hash,)
    ).fetchone()

    if existing is not None:
        image_id = existing[0]
    else:
        archived = image_pipeline.archive_image(prepared, archive_root)
        analysis = image_pipeline.analyze(
            vision_provider, prepared, model=vision_model,
            prefetched_ocr=route.prefetched_ocr,
        )
        cur.execute(
            "INSERT INTO images "
            "(file_hash, file_path, width, height, analysis_json, analysis_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (prepared.hash, str(archived), prepared.width, prepared.height,
             analysis.model_dump_json(), vision_model),
        )
        image_id = conn.last_insert_rowid()
        chunk_inputs = image_pipeline.analysis_to_chunk_inputs(analysis)
        if chunk_inputs:
            insert_image_chunks(conn, image_id, chunk_inputs)

    # OR IGNORE: re-ingest after delete can legitimately replay the same
    # (doc, image, page, index) tuple; no-op rather than crash the whole file.
    cur.execute(
        "INSERT OR IGNORE INTO document_images "
        "(document_id, image_id, page_number, image_index_on_page) "
        "VALUES (?, ?, ?, ?)",
        (document_id, image_id, route.page_number, route.image_index_on_page),
    )


# -------------------- per-file dispatch --------------------


def _ingest_text_document(
    conn,
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    """txt / html / md via the existing convert_and_chunk path."""
    if on_stage is not None:
        on_stage("extracting")
    result = convert_and_chunk(archived)
    document_id = _insert_document(
        conn,
        file_hash=file_hash,
        file_name=file_name,
        archive_path=archived,
        page_count=result.page_count,
        full_text=result.full_text,
    )
    if result.chunks:
        embeddings = embed_texts([row.text for row in result.chunks])
        insert_document_chunks(
            conn, document_id, _build_chunk_inputs(result.chunks, embeddings),
        )
    return document_id, result.full_text


def _sec2md_chunk_to_row(chunk, *, fallback_heading: str | None = None) -> ChunkRow:
    """Map a ``Sec2mdChunk`` to a ``ChunkRow``, falling back to ``fallback_heading``
    when sec2md didn't detect a section header for the chunk."""
    return ChunkRow(
        text=chunk.text,
        section_heading=chunk.section_heading or fallback_heading,
        content_type=chunk.content_type,
        page_number=chunk.page_number,
    )


def _ingest_html_sec2md(
    conn,
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    """iXBRL EDGAR filing via sec2md."""
    if on_stage is not None:
        on_stage("extracting")
    result = sec2md_pipeline.convert(archived)
    document_id = _insert_document(
        conn,
        file_hash=file_hash,
        file_name=file_name,
        archive_path=archived,
        page_count=result.page_count,
        full_text=result.full_text,
    )
    if result.chunks:
        if on_stage is not None:
            on_stage("embedding")
        rows = [_sec2md_chunk_to_row(c) for c in result.chunks]
        embeddings = embed_texts([r.text for r in rows])
        insert_document_chunks(
            conn, document_id, _build_chunk_inputs(rows, embeddings),
        )
    return document_id, result.full_text


def _ingest_edgar_submission(
    conn,
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[int, str]:
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
    document_id = _insert_document(
        conn,
        file_hash=file_hash,
        file_name=file_name,
        archive_path=archived,
        page_count=None,
        full_text=full_text,
    )
    if rows:
        if on_stage is not None:
            on_stage("embedding")
        embeddings = embed_texts([r.text for r in rows])
        insert_document_chunks(
            conn, document_id, _build_chunk_inputs(rows, embeddings),
        )
    return document_id, full_text


def _ingest_pdf_pdfplumber(
    conn,
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    archive_root: Path,
    sparse_text_threshold: int,
    ocr_min_confidence: int,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_max_dimension: int,
    vision_min_dimension: int,
    on_page_progress: Callable[[int, int], None] | None = None,
    on_image_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_page_count: Callable[[int | None], None] | None = None,
) -> tuple[int, str]:
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
    document_id = _insert_document(
        conn,
        file_hash=file_hash,
        file_name=file_name,
        archive_path=archived,
        page_count=result.page_count,
        full_text=result.full_text,
    )

    doc_chunks: list[ChunkRow] = []
    image_routes: list[_ImageRoute] = []

    for page in result.pages:
        if page.content_type is not None:
            for piece in chunk_text(page.text):
                doc_chunks.append(ChunkRow(
                    text=piece,
                    section_heading=None,
                    content_type=page.content_type,
                    page_number=page.page_number,
                ))
        elif page.page_render_png is not None:
            # Sparse page where OCR didn't clear the bar — fall back to VLM.
            # Page-level Tesseract already ran; hand the result down so the
            # image pipeline doesn't re-OCR the same bytes.
            image_routes.append(_ImageRoute(
                bytes_=page.page_render_png,
                page_number=page.page_number,
                image_index_on_page=0,
                prefetched_ocr=page.ocr_result,
            ))

        for emb in page.embedded_images:
            image_routes.append(_ImageRoute(
                bytes_=emb.png_bytes,
                page_number=page.page_number,
                image_index_on_page=emb.image_index_on_page,
            ))

    if on_stage is not None:
        on_stage("embedding")
    if doc_chunks:
        embeddings = embed_texts([r.text for r in doc_chunks])
        insert_document_chunks(
            conn, document_id, _build_chunk_inputs(doc_chunks, embeddings),
        )

    if on_stage is not None and image_routes:
        on_stage("analyzing images")
    _run_image_routes(
        conn, document_id, image_routes,
        archive_root=archive_root,
        vision_provider=vision_provider,
        vision_model=vision_model,
        vision_max_dimension=vision_max_dimension,
        vision_min_dimension=vision_min_dimension,
        on_progress=on_image_progress,
    )

    return document_id, result.full_text


def _ingest_pdf_docling(
    conn,
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    archive_root: Path,
    sparse_text_threshold: int,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_max_dimension: int,
    vision_min_dimension: int,
    on_image_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_page_count: Callable[[int | None], None] | None = None,
) -> tuple[int, str]:
    from bartleby.ingest import docling as docling_pipeline

    if on_stage is not None:
        on_stage("extracting")
    docling_result = docling_pipeline.convert(archived)
    if on_page_count is not None:
        on_page_count(docling_result.page_count)
    document_id = _insert_document(
        conn,
        file_hash=file_hash,
        file_name=file_name,
        archive_path=archived,
        page_count=docling_result.page_count,
        full_text=docling_result.full_text,
    )
    if docling_result.chunks:
        if on_stage is not None:
            on_stage("embedding")
        rows = [
            ChunkRow(text=c.text, section_heading=c.section_heading,
                     content_type=c.content_type, page_number=c.page_number)
            for c in docling_result.chunks
        ]
        embeddings = embed_texts([r.text for r in rows])
        insert_document_chunks(
            conn, document_id, _build_chunk_inputs(rows, embeddings),
        )

    # Side-pass for embedded images so docling users still get image search.
    if vision_provider is not None:
        if on_stage is not None:
            on_stage("analyzing images")
        pdf_result = pdfplumber_pipeline.convert(
            archived,
            sparse_text_threshold=sparse_text_threshold,
            ocr_min_confidence=0,    # text discarded; OCR doesn't matter
        )
        image_routes = [
            _ImageRoute(
                bytes_=emb.png_bytes,
                page_number=page.page_number,
                image_index_on_page=emb.image_index_on_page,
            )
            for page in pdf_result.pages
            for emb in page.embedded_images
        ]
        _run_image_routes(
            conn, document_id, image_routes,
            archive_root=archive_root,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_max_dimension=vision_max_dimension,
            vision_min_dimension=vision_min_dimension,
            on_progress=on_image_progress,
        )

    return document_id, docling_result.full_text


def _ingest_image_file(
    conn,
    archived: Path,
    *,
    file_hash: str,
    file_name: str,
    archive_root: Path,
    vision_provider: Provider,
    vision_model: str,
    vision_max_dimension: int,
    vision_min_dimension: int,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    if on_stage is not None:
        on_stage("analyzing image")
    document_id = _insert_document(
        conn,
        file_hash=file_hash,
        file_name=file_name,
        archive_path=archived,
        page_count=None,
        full_text="",
    )
    route = _ImageRoute(
        bytes_=archived.read_bytes(),
        page_number=None,
        image_index_on_page=0,
    )
    _process_image(
        conn, document_id, route,
        archive_root=archive_root,
        vision_provider=vision_provider,
        vision_model=vision_model,
        vision_max_dimension=vision_max_dimension,
        vision_min_dimension=vision_min_dimension,
    )
    # No full_text to return: a standalone image has no document chunks, so the
    # summarizer builds its input from the image chunks directly
    # (_build_summary_input). The fallback_text is never reached here.
    return document_id, ""


def _run_image_routes(
    conn,
    document_id: int,
    routes: list[_ImageRoute],
    *,
    archive_root: Path,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_max_dimension: int,
    vision_min_dimension: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    if not routes:
        return
    if vision_provider is None or not vision_model:
        console.warn(
            f"Skipping {len(routes)} image(s) — no vision provider configured."
        )
        return
    total = len(routes)
    if on_progress is not None:
        on_progress(0, total)
    for i, route in enumerate(routes, start=1):
        try:
            _process_image(
                conn, document_id, route,
                archive_root=archive_root,
                vision_provider=vision_provider,
                vision_model=vision_model,
                vision_max_dimension=vision_max_dimension,
                vision_min_dimension=vision_min_dimension,
            )
        except Exception as e:
            page_str = f" (page {route.page_number})" if route.page_number else ""
            console.warn(f"Image analysis failed{page_str}: {e}")
        finally:
            if on_progress is not None:
                on_progress(i, total)


# -------------------- top-level orchestrator --------------------


HTML_EXTENSIONS = {".html", ".htm"}


def _process_one(
    conn,
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
) -> bool:
    """Returns True if a new document was ingested, False if the file was
    already ingested and skipped. Caller handles the user-facing message."""
    file_hash = _hash_file(path)
    if _document_already_ingested(conn, file_hash) is not None:
        return False

    # If a stranded partial row exists for this hash (interrupted run), drop
    # it so the upcoming INSERT doesn't trip the file_hash UNIQUE constraint.
    stranded = conn.cursor().execute(
        "SELECT document_id FROM documents WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    if stranded is not None:
        _cleanup_partial_document(conn, stranded[0])

    archived = _archive(path, archive_root, file_hash, ext)

    if ext in IMAGE_EXTENSIONS:
        if vision_provider is None:
            console.warn(
                f"{path.name}: skipping image (no vision provider configured)."
            )
            return
        document_id, full_text = _ingest_image_file(
            conn, archived,
            file_hash=file_hash, file_name=path.name,
            archive_root=archive_root,
            vision_provider=vision_provider, vision_model=vision_model,
            vision_max_dimension=vision_max_dimension,
            vision_min_dimension=vision_min_dimension,
            on_stage=on_stage,
        )
    elif ext in PDF_EXTENSIONS:
        if pdf_converter == "docling":
            document_id, full_text = _ingest_pdf_docling(
                conn, archived,
                file_hash=file_hash, file_name=path.name,
                archive_root=archive_root,
                sparse_text_threshold=sparse_text_threshold,
                vision_provider=vision_provider, vision_model=vision_model,
                vision_max_dimension=vision_max_dimension,
                vision_min_dimension=vision_min_dimension,
                on_image_progress=on_image_progress,
                on_stage=on_stage,
                on_page_count=on_page_count,
            )
        else:
            document_id, full_text = _ingest_pdf_pdfplumber(
                conn, archived,
                file_hash=file_hash, file_name=path.name,
                archive_root=archive_root,
                sparse_text_threshold=sparse_text_threshold,
                ocr_min_confidence=ocr_min_confidence,
                vision_provider=vision_provider, vision_model=vision_model,
                vision_max_dimension=vision_max_dimension,
                vision_min_dimension=vision_min_dimension,
                on_page_progress=on_page_progress,
                on_image_progress=on_image_progress,
                on_stage=on_stage,
                on_page_count=on_page_count,
            )
    elif edgar_pipeline.detect(archived):
        # EDGAR full-submission SGML envelope — detected by content, so a `.txt`
        # wrapper is caught here instead of falling to the character chunker.
        document_id, full_text = _ingest_edgar_submission(
            conn, archived,
            file_hash=file_hash, file_name=path.name,
            on_stage=on_stage,
        )
    elif ext in HTML_EXTENSIONS and html_converter == "sec2md" and sec2md_pipeline.is_ixbrl(archived):
        document_id, full_text = _ingest_html_sec2md(
            conn, archived,
            file_hash=file_hash, file_name=path.name,
            on_stage=on_stage,
        )
    else:
        document_id, full_text = _ingest_text_document(
            conn, archived, file_hash=file_hash, file_name=path.name,
            on_stage=on_stage,
        )

    _maybe_summarize(
        conn, document_id, full_text,
        provider=llm_provider, model=llm_model,
        temperature=temperature, max_summarize_tokens=max_summarize_tokens,
        on_stage=on_stage,
    )
    return True


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

            for src, src_ext in sources:
                # Render the main row from the active stage + page count. The
                # filename is truncated so a long name can't squeeze the bar
                # (issue #85); the page count is appended once extraction knows
                # it, for whichever converter ran.
                state = {"stage": "starting", "pages": None}

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

                def _set_stage(stage: str, _state=state, _render=_render) -> None:
                    _state["stage"] = stage
                    _render()

                def _set_page_count(count: int | None, _state=state, _render=_render) -> None:
                    _state["pages"] = count
                    _render()

                _render()
                try:
                    was_ingested = _process_one(
                        conn, src, src_ext, archive_root,
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
                    if not was_ingested:
                        skipped_count += 1
                        if skipped_count <= SKIP_DISPLAY_LIMIT:
                            console.info(
                                f"Skipping {src.name} (already ingested)"
                            )
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
    finally:
        conn.close()

    console.complete("Done.")
