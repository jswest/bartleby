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

import json
import sys
import time
from pathlib import Path

from loguru import logger

from bartleby.config import (
    config_drift,
    load_config,
    redact_config,
)
from bartleby.db.connection import open_db, resolve_project_name
from bartleby.ingest import caption
from bartleby.ingest import classify
from bartleby.ingest import parse
from bartleby.ingest import parsers
from bartleby.ingest import resolve
from bartleby.ingest import summary
from bartleby.ingest.progress import ScribeProgress
from bartleby.ingest.chunk import resolve_format_filter
from bartleby.ingest.writer import FailedUnit, MAX_INGEST_ATTEMPTS, Writer
from bartleby.lib import console
from bartleby.lib import timing
from bartleby.lib.consts import (
    ALLOWED_HTML_CONVERTERS,
    ALLOWED_PDF_CONVERTERS,
    DEFAULT_HTML_CONVERTER,
    DEFAULT_MAX_SUMMARIZE_TOKENS,
    DEFAULT_OCR_MIN_CONFIDENCE,
    DEFAULT_PDF_CONVERTER,
    DEFAULT_SPARSE_TEXT_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_VISION_MAX_DIMENSION,
    DEFAULT_VISION_MIN_DIMENSION,
)
from bartleby.project import get_project_dir


def _report_failures(failures: list[FailedUnit]) -> None:
    """Surface every still-unresolved ingest unit so a skipped one never reads
    as a green run. Capped units won't be retried; the rest resume next run."""
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
    if pdf_converter_name not in ALLOWED_PDF_CONVERTERS:
        raise ValueError(
            f"Unknown pdf_converter {pdf_converter_name!r}; "
            f"expected one of {', '.join(ALLOWED_PDF_CONVERTERS)}."
        )
    html_converter_name = (
        html_converter or config.get("html_converter", DEFAULT_HTML_CONVERTER)
    ).lower()
    if html_converter_name not in ALLOWED_HTML_CONVERTERS:
        raise ValueError(
            f"Unknown html_converter {html_converter_name!r}; "
            f"expected one of {', '.join(ALLOWED_HTML_CONVERTERS)}."
        )

    required_models = resolve._required_hf_models(
        pdf_converter_name, html_converter_name
    )
    from bartleby.lib.quiet import setup_quiet_third_party
    setup_quiet_third_party(verbose=verbose, required_models=required_models)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    project_name = resolve_project_name(project)

    # Fold the --provider override in once; both the config snapshot and the
    # #314 summarize-worker clamp key off this, never bare config['provider'].
    effective_provider = provider or config.get("provider")
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

    sources, unidentified = classify._collect_files(
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
    temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
    reasoning_effort = config.get("reasoning_effort")
    max_summarize_tokens = int(
        config.get("max_summarize_tokens", DEFAULT_MAX_SUMMARIZE_TOKENS)
    )
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
            "provider": effective_provider,
            "model": model or config.get("model"),
        })
        for line in config_drift(writer.latest_config(), config_snapshot):
            console.warn(f"Config drift since last ingest — {line}")
        writer.begin_run(config_snapshot)

        # Classify up front (hash + resume lookups, on this process): only files
        # that have never been parsed cross to the workers. Parsed-but-uncaptioned
        # docs resume on the main process — they need no parse, just the missing
        # captions. Summaries are settled afterwards by their own pass.
        to_parse, to_resume, skipped, duplicates = classify._classify(
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
        summarize_workers = resolve._resolve_summarize_workers(
            config, effective_provider=effective_provider, timings=timings
        )
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
            # Drains the pool through the single Writer and returns the DocUnits
            # (resumed + freshly parsed) the next two phases carry forward.
            incomplete_count = 0
            units = parse.parse_all(
                writer, to_parse, to_resume,
                parse_config=parse_config,
                max_workers=max_workers,
                progress=progress,
                required_models=required_models,
                verbose=verbose,
                timings=timings,
            )

            # ---- Phase 2: caption every uncaptioned image, concurrently ------
            # _caption_all drives the tally via on_progress (0,total then per
            # image) and the lanes via on_lane (per worker thread); the phase
            # adapts the tally contract itself and reveals the caption total the
            # first time it hears one.
            cap_phase = progress.phase("caption")
            caption._caption_all(
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
                    owed, summarize_times = summary._summarize_all(
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

        failures = writer.failures()
        _report_failures(failures)
        # The exit code is a scripted caller's only signal: a run that left any
        # unit unresolved (a parse failure recorded in failed_ingests, or a
        # document still owing captions/summary) must not read as green. Capture
        # before `finally` closes the connection.
        had_failures = incomplete_count > 0 or bool(failures)
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

    if had_failures:
        # Warnings above already named what failed; skip the green "Done." and
        # exit non-zero so `bartleby scribe ... && next-step` halts the chain.
        sys.exit(1)
    console.complete("Done.")
