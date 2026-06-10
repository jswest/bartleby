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
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

from bartleby.config import (
    config_drift,
    load_config,
    redact_config,
)
from bartleby.db.chunks import ChunkInput
from bartleby.db.connection import open_db, resolve_project_name
from bartleby.ingest import images as image_pipeline
from bartleby.ingest import parsers
from bartleby.ingest import pool
from bartleby.ingest import resolve
from bartleby.ingest.progress import ScribeProgress, _ProgressTally
from bartleby.ingest.chunk import (
    IMAGE_EXTENSIONS,
    chunk_markdown_string,
    resolve_extension,
    resolve_format_filter,
)
from bartleby.ingest.embed import embed_texts
from bartleby.ingest.summarize import summarize
from bartleby.ingest.writer import (
    MAX_INGEST_ATTEMPTS,
    ImageCaption,
    PendingImage,
    PendingSummary,
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
)
from bartleby.project import get_project_dir
from bartleby.providers import Provider


# -------------------- shared helpers --------------------


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def _summary_chunks(text: str) -> list[ChunkInput]:
    """Chunk + embed a summary's markdown body into ChunkInputs (producer side)."""
    rows = chunk_markdown_string(text)
    if not rows:
        return []
    embeddings = embed_texts([r.text for r in rows])
    return parsers._build_chunk_inputs(rows, embeddings)


# -------------------- image caption helpers --------------------


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
    on_lane: Callable[[object, str, str], None] | None = None,
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

    progress = _ProgressTally(len(pending), on_progress)

    # Capped rows (failed MAX_INGEST_ATTEMPTS× already) are skipped, not retried.
    to_caption: dict[int, PendingImage] = {}
    for image_id, pi in pending.items():
        if writer.is_capped(pi.file_hash, "caption"):
            console.warn(
                f"{owner[image_id].file_name}: skipping image — failed "
                f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
            )
            progress.advance()
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

    def _analyze(image_id: int, pi: PendingImage) -> image_pipeline.ImageAnalysis:
        # Claim this thread's lane for the owning document, then run the analysis.
        # Keyed by thread id, so the pool's N threads map to N sticky lanes.
        if on_lane is not None:
            on_lane(threading.get_ident(), owner[image_id].file_name, "captioning")
        return _analyze_image(
            pi, vision_provider=vision_provider, vision_model=vision_model,
        )

    if caption_workers <= 1:
        # Inline: one image at a time on this thread. Same analyze/persist calls
        # as the pool, but with clean per-document caption seconds for --timings.
        for image_id, pi in to_caption.items():
            t0 = time.perf_counter() if timings else None
            try:
                _persist(image_id, _analyze(image_id, pi))
            except Exception as e:
                _fail(image_id, e)
            unit = owner[image_id]
            if t0 is not None and unit.stages is not None:
                unit.stages["caption"] = (
                    unit.stages.get("caption", 0.0) + (time.perf_counter() - t0)
                )
            progress.advance()
        return

    with ThreadPoolExecutor(max_workers=caption_workers) as pool:
        futures = {
            pool.submit(_analyze, image_id, pi): image_id
            for image_id, pi in to_caption.items()
        }
        for fut in as_completed(futures):
            image_id = futures[fut]
            try:
                _persist(image_id, fut.result())
            except Exception as e:
                _fail(image_id, e)
            progress.advance()


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
    on_lane: Callable[[object, str, str], None] | None = None,
    reasoning_effort: str | None = None,
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
    incomplete = 0
    times: dict[int, tuple[str, float]] = {}
    progress = _ProgressTally(len(pending), on_progress)

    # Capped docs (failed MAX_INGEST_ATTEMPTS× already) are surfaced, not retried.
    work: list[PendingSummary] = []
    for ps in pending:
        if writer.is_capped(ps.file_hash, "summary"):
            console.warn(
                f"{ps.file_name}: skipping summary — failed "
                f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
            )
            incomplete += 1
            progress.advance()
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

    def _summarize(ps: PendingSummary, text: str):
        # Claim the running thread's lane for this document, then make the LLM
        # call — keyed by thread id so the pool's N threads map to N sticky lanes.
        if on_lane is not None:
            on_lane(threading.get_ident(), ps.file_name, "summarizing")
        return summarize(
            text, provider=llm_provider, model=llm_model,
            temperature=temperature, max_summarize_tokens=max_summarize_tokens,
            reasoning_effort=reasoning_effort,
        )

    if summarize_workers <= 1:
        # Inline: one document at a time on this thread. Same summarize/persist
        # calls as the pool, but with clean per-document seconds for --timings.
        for ps in work:
            t0 = time.perf_counter() if timings else None
            try:
                _persist(ps, _summarize(ps, writer.summary_input(ps.document_id)))
            except Exception as e:
                _fail(ps, e)
            if t0 is not None:
                times[ps.document_id] = (ps.file_name, time.perf_counter() - t0)
            progress.advance()
        return incomplete, times

    # Pooled: keep at most ``summarize_workers`` inputs in flight (see docstring) —
    # fetch each document's text on this thread, then top up one as each completes.
    it = iter(work)
    with ThreadPoolExecutor(max_workers=summarize_workers) as pool:
        in_flight = {
            pool.submit(_summarize, ps, writer.summary_input(ps.document_id)): ps
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
                progress.advance()
                nxt = next(it, None)
                if nxt is not None:
                    in_flight[
                        pool.submit(
                            _summarize, nxt, writer.summary_input(nxt.document_id)
                        )
                    ] = nxt
    return incomplete, times


# -------------------- parse pool: requests, config, worker fn --------------------


@dataclass
class _ResumeItem:
    """A document parsed by an earlier run that still has missing units."""
    document_id: int
    file_name: str
    file_hash: str


def _classify(
    writer: Writer,
    sources: list[tuple[Path, str]],
    *,
    vision_enabled: bool,
) -> tuple[list[parsers.ParseRequest], list[_ResumeItem], list[str], list[str]]:
    """Bucket each source by what the parse/caption drain still needs, all from
    DB state: already parsed+captioned (skip), parsed-but-uncaptioned (resume —
    no parse), or never parsed (hand to the pool). Summaries are not considered
    here — they're settled by their own pass over whatever the DB still lacks.
    Hashing + the resume lookups run here on the main process; only the parse
    bucket crosses to workers.

    A fourth bucket, ``duplicates``, catches two byte-identical files *within one
    run*: ``documents.file_hash`` is UNIQUE, so only the first can persist. The
    DB lookup can't see the in-run twin (neither is committed yet), so we track
    hashes queued this run and divert the rest here rather than let the second
    crash ``persist_parse`` (#225)."""
    to_parse: list[parsers.ParseRequest] = []
    to_resume: list[_ResumeItem] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    queued_hashes: set[str] = set()
    for path, ext in sources:
        file_hash = _hash_file(path)
        document_id = writer.document_id_for(file_hash)
        if document_id is not None:
            if _is_complete(writer, document_id):
                skipped.append(path.name)
            else:
                to_resume.append(_ResumeItem(document_id, path.name, file_hash))
            continue
        if file_hash in queued_hashes:
            duplicates.append(path.name)
            continue
        if ext in IMAGE_EXTENSIONS and not vision_enabled:
            console.warn(
                f"{path.name}: skipping image (no vision provider configured)."
            )
            continue
        queued_hashes.add(file_hash)
        to_parse.append(
            parsers.ParseRequest(
                path=path, ext=ext, file_hash=file_hash, file_name=path.name
            )
        )
    return to_parse, to_resume, skipped, duplicates


# -------------------- entry --------------------


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
        required_models=resolve._required_hf_models(
            pdf_converter_name, html_converter_name
        ),
    )

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    project_name = resolve_project_name(project)

    llm_provider, llm_model = resolve._resolve_llm_provider(
        config, provider_override=provider, model_override=model,
    )
    vision_provider, vision_model = resolve._resolve_vision_provider(config)

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

    parse_config = parsers.ParseConfig(
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
    reasoning_effort = config.get("reasoning_effort")
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
        to_parse, to_resume, skipped, duplicates = _classify(
            writer, sources,
            vision_enabled=parse_config.vision_enabled,
        )
        SKIP_DISPLAY_LIMIT = 3
        for names, reason in ((skipped, "already ingested"),
                              (duplicates, "duplicate content within this run")):
            for name in names[:SKIP_DISPLAY_LIMIT]:
                console.info(f"Skipping {name} ({reason})")
            if len(names) > SKIP_DISPLAY_LIMIT:
                console.info(
                    f"… and {len(names) - SKIP_DISPLAY_LIMIT} more file(s) "
                    f"skipped as {reason}"
                )

        max_workers = (
            resolve._resolve_max_workers(config, timings=timings) if to_parse else 1
        )
        caption_workers = resolve._resolve_caption_workers(config, timings=timings)
        summarize_workers = resolve._resolve_summarize_workers(config, timings=timings)
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
        #
        # ScribeProgress renders all three at once (#170): a run-of-show header
        # tallying the phases, one overall bar over every pipeline unit, and one
        # lane per worker. It shares the console module's Rich Console, so
        # console.info/error during the run insert above the live display rather
        # than colliding with it (and stdout-redirect captures around Docling/
        # RapidOCR don't break it). Lanes ≥ the widest phase's worker count.
        n_lanes = max(max_workers, caption_workers, summarize_workers)
        with ScribeProgress(n_lanes=n_lanes) as progress:
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
            parse_phase = progress.phase("parse")
            parse_phase.start(len(to_parse))
            for outcome in pool.parse_stream(
                to_parse,
                parse_fn=parsers._parse_request,
                config=parse_config,
                max_workers=max_workers,
                warmup=parsers._warm_worker,
                verbose=verbose,
                required_models=resolve._required_hf_models(
                    pdf_converter_name, html_converter_name
                ),
                on_progress=lambda ev: parse_phase.lane(ev.worker, ev.item, ev.stage),
            ):
                req = outcome.request
                # Notices the worker collected (it has no Live console of its
                # own); emit them here, where console.warn coordinates with the
                # progress bar instead of stomping it.
                for warning in outcome.warnings:
                    console.warn(warning)
                if outcome.error is not None:
                    writer.record_failure(
                        req.file_hash, req.file_name, "parse", outcome.error
                    )
                    message = f"Failed: {req.file_name}: {outcome.error}"
                    if outcome.offline:
                        from bartleby.lib.quiet import OFFLINE_HINT
                        message += f"\n  {OFFLINE_HINT}"
                    console.error(message)
                    parse_phase.advance()
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
                parse_phase.advance()

            # ---- Phase 2: caption every uncaptioned image, concurrently ------
            # _caption_all drives the tally via on_progress (0,total then per
            # image) and the lanes via on_lane (per worker thread); the phase
            # adapts the tally contract itself and reveals the caption total the
            # first time it hears one.
            cap_phase = progress.phase("caption")
            _caption_all(
                writer, units,
                vision_provider=vision_provider, vision_model=vision_model,
                vision_enabled=parse_config.vision_enabled,
                caption_workers=caption_workers, timings=timings,
                on_progress=cap_phase.on_progress,
                on_lane=cap_phase.lane,
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
                    sum_phase = progress.phase("summarize")
                    owed, summarize_times = _summarize_all(
                        writer, pending,
                        llm_provider=llm_provider, llm_model=llm_model,
                        temperature=temperature,
                        max_summarize_tokens=max_summarize_tokens,
                        summarize_workers=summarize_workers, timings=timings,
                        on_progress=sum_phase.on_progress,
                        on_lane=sum_phase.lane,
                        reasoning_effort=reasoning_effort,
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
