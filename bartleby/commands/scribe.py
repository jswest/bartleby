"""`bartleby scribe` — sequential ingest pipeline.

Hash → archive → convert → embed → (optionally) summarize → write. All chunk
writes go through ``bartleby.db.chunks`` typed helpers; image rows additionally
get a join entry in ``document_images``.

PDF path is backend-aware (``pdfplumber`` default, ``docling`` opt-in). Image
files (jpg/png/etc.) go straight through the VLM pipeline. txt/html/md keep
the original ``convert_and_chunk`` path.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger
from rich.console import Console
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
from bartleby.db.connection import open_db
from bartleby.ingest import images as image_pipeline
from bartleby.ingest import ocr as ocr_module
from bartleby.ingest import pdfplumber as pdfplumber_pipeline
from bartleby.ingest.chunk import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    ChunkRow,
    chunk_markdown_string,
    convert_and_chunk,
)
from bartleby.ingest.embed import embed_texts
from bartleby.ingest.summarize import SummaryResult, count_tokens, summarize
from bartleby.ingest.text import chunk_text
from bartleby.lib import console
from bartleby.project import get_active_project, get_project_dir
from bartleby.providers import Provider, get_provider


DEFAULT_BACKEND = "pdfplumber"
DEFAULT_SPARSE_TEXT_THRESHOLD = 100
DEFAULT_OCR_MIN_CONFIDENCE = 30
DEFAULT_VISION_MAX_DIMENSION = 1024


# -------------------- shared helpers --------------------


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _archive(src: Path, archive_root: Path, file_hash: str) -> Path:
    ext = src.suffix.lower()
    dest_dir = archive_root / file_hash
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{file_hash}{ext}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise ValueError(f"Path not found: {path}")


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
        "(document_id, title, description, text, model) "
        "VALUES (?, ?, ?, ?, ?)",
        (document_id, summary.title, summary.description,
         summary.text, summary.model),
    )
    summary_id = conn.last_insert_rowid()

    rows = chunk_markdown_string(summary.text)
    if rows:
        embeddings = embed_texts([r.text for r in rows])
        insert_summary_chunks(conn, summary_id, _build_chunk_inputs(rows, embeddings))
    return summary_id


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
    if provider is None or not model or not full_text or not full_text.strip():
        return
    if on_stage is not None:
        on_stage("summarizing")
    result = summarize(
        full_text,
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
) -> None:
    """Run the §7 image pipeline on one image and persist it.

    Dedupes on the image's source-byte hash: if we have already analyzed an
    identical image (across any document), reuse the ``images`` row and add a
    new ``document_images`` join. Chunks come along automatically since they
    point at ``image_id``.
    """
    prepared = image_pipeline.prepare_image(
        route.bytes_, max_dimension=vision_max_dimension,
    )
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
    on_page_progress: Callable[[int, int], None] | None = None,
    on_image_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    if on_stage is not None:
        on_stage("extracting")
    result = pdfplumber_pipeline.convert(
        archived,
        sparse_text_threshold=sparse_text_threshold,
        ocr_min_confidence=ocr_min_confidence,
        on_progress=on_page_progress,
    )
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
    on_image_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    from bartleby.ingest import docling as docling_pipeline

    if on_stage is not None:
        on_stage("extracting")
    docling_result = docling_pipeline.convert(archived)
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
                     content_type=c.content_type)
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
    )
    # For summarization purposes, hand the LLM the concatenated chunk text.
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT c.text FROM chunks c "
        "JOIN images i ON i.image_id = c.source_id "
        "JOIN document_images di ON di.image_id = i.image_id "
        "WHERE c.source_kind = 'image' AND di.document_id = ? "
        "ORDER BY c.chunk_index",
        (document_id,),
    )
    full_text = "\n\n".join(r[0] for r in rows)
    return document_id, full_text


def _run_image_routes(
    conn,
    document_id: int,
    routes: list[_ImageRoute],
    *,
    archive_root: Path,
    vision_provider: Provider | None,
    vision_model: str | None,
    vision_max_dimension: int,
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
            )
        except Exception as e:
            page_str = f" (page {route.page_number})" if route.page_number else ""
            console.warn(f"Image analysis failed{page_str}: {e}")
        finally:
            if on_progress is not None:
                on_progress(i, total)


# -------------------- top-level orchestrator --------------------


def _process_one(
    conn,
    path: Path,
    archive_root: Path,
    *,
    backend: str,
    sparse_text_threshold: int,
    ocr_min_confidence: int,
    vision_max_dimension: int,
    llm_provider: Provider | None,
    llm_model: str | None,
    temperature: float,
    max_summarize_tokens: int,
    vision_provider: Provider | None,
    vision_model: str | None,
    on_page_progress: Callable[[int, int], None] | None = None,
    on_image_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
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

    archived = _archive(path, archive_root, file_hash)

    ext = path.suffix.lower()
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
            on_stage=on_stage,
        )
    elif ext in PDF_EXTENSIONS:
        if backend == "docling":
            document_id, full_text = _ingest_pdf_docling(
                conn, archived,
                file_hash=file_hash, file_name=path.name,
                archive_root=archive_root,
                sparse_text_threshold=sparse_text_threshold,
                vision_provider=vision_provider, vision_model=vision_model,
                vision_max_dimension=vision_max_dimension,
                on_image_progress=on_image_progress,
                on_stage=on_stage,
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
                on_page_progress=on_page_progress,
                on_image_progress=on_image_progress,
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


def main(
    *,
    project: str | None,
    files: str | Path,
    model: str | None = None,
    provider: str | None = None,
    backend: str | None = None,
    verbose: bool = False,
) -> None:
    from bartleby.lib.quiet import setup_quiet_third_party
    setup_quiet_third_party(verbose=verbose)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    project_name = project or get_active_project()
    if not project_name:
        raise RuntimeError(
            "No active project. Run `bartleby project create <name>` first."
        )

    config = load_config()
    llm_provider, llm_model = _resolve_llm_provider(
        config, provider_override=provider, model_override=model,
    )
    vision_provider, vision_model = _resolve_vision_provider(config)
    backend_name = (backend or config.get("backend", DEFAULT_BACKEND)).lower()
    if backend_name not in ("pdfplumber", "docling"):
        raise ValueError(
            f"Unknown backend {backend_name!r}; expected 'pdfplumber' or 'docling'."
        )

    sources = _collect_files(Path(files))
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

            for src in sources:
                def _set_stage(stage: str, _src=src) -> None:
                    bar.update(task, label=f"Ingesting {_src.name} · {stage}")

                _set_stage("starting")
                try:
                    was_ingested = _process_one(
                        conn, src, archive_root,
                        backend=backend_name,
                        sparse_text_threshold=sparse_text_threshold,
                        ocr_min_confidence=ocr_min_confidence,
                        vision_max_dimension=vision_max_dimension,
                        llm_provider=llm_provider, llm_model=llm_model,
                        temperature=temperature,
                        max_summarize_tokens=max_summarize_tokens,
                        vision_provider=vision_provider,
                        vision_model=vision_model,
                        on_page_progress=_phase_callback(f"  pages ({src.name})"),
                        on_image_progress=_phase_callback(f"  images ({src.name})"),
                        on_stage=_set_stage,
                    )
                    if not was_ingested:
                        skipped_count += 1
                        if skipped_count <= SKIP_DISPLAY_LIMIT:
                            console.info(
                                f"Skipping {src.name} (already ingested)"
                            )
                except Exception as e:
                    console.error(f"Failed: {src.name}: {e}")
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
