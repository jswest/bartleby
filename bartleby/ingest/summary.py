"""Phase 3 of the scribe ingest: summarize every document still owing one.

Pulled out of ``commands/scribe.py`` (#306), and kept separate from the existing
``summarize.py`` (which it calls) to avoid a writer<->summarize import cycle.
Mirrors the caption stage (#166, #188): the network-bound ``summarize()`` LLM
call fans out across summarize workers while the Writer's persist stays on the
calling thread.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from bartleby.db.chunks import ChunkInput
from bartleby.ingest import embed
from bartleby.ingest import parsers
from bartleby.ingest.chunk import chunk_markdown_string
from bartleby.ingest.progress import _ProgressTally
from bartleby.ingest.summarize import summarize
from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS, PendingSummary, Writer
from bartleby.lib import console
from bartleby.providers import Provider


def _summary_chunks(text: str) -> list[ChunkInput]:
    """Chunk + embed a summary's markdown body into ChunkInputs (producer side)."""
    rows = chunk_markdown_string(text)
    if not rows:
        return []
    embeddings = embed.embed_texts([r.text for r in rows])
    return parsers._build_chunk_inputs(rows, embeddings)

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
