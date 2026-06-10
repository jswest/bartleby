"""Phase 1 of the scribe ingest: parse + persist every queued file.

Pulled out of ``commands/scribe.py`` (#306). ``parse_all`` fans the DB-free
parse out across worker *processes* (or runs inline at ``max_workers <= 1``) and
drains each :class:`~bartleby.ingest.parsers.ParseOutcome` back through the
single Writer on the calling thread. It also owns :class:`DocUnit`, the
per-document unit the caption and summarize phases carry forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bartleby.ingest import parsers
from bartleby.ingest import pool
from bartleby.ingest.progress import ScribeProgress
from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS, Writer
from bartleby.lib import console

if TYPE_CHECKING:
    from bartleby.ingest.classify import _ResumeItem


@dataclass
class DocUnit:
    """A document parsed + persisted this run (or resumed from an earlier one),
    carried through the caption and summarize phases. ``stages`` accumulates
    per-stage seconds only under ``--timings`` — it's None on the fast path."""
    document_id: int
    file_name: str
    file_hash: str
    page_count: int | None = None
    stages: dict[str, float] | None = None


def parse_all(
    writer: Writer,
    to_parse: list[parsers.ParseRequest],
    to_resume: list[_ResumeItem],
    *,
    parse_config: parsers.ParseConfig,
    max_workers: int,
    progress: ScribeProgress,
    required_models: tuple[str, ...],
    verbose: bool,
    timings: bool,
) -> list[DocUnit]:
    """Drain the parse phase, returning the DocUnits the later phases carry.

    Resumed docs were parsed by an earlier run — they need no parse, just the
    missing captions the caption phase fills; summaries are settled by their own
    pass over the DB (#167). Fresh files cross to the pool; the single Writer
    persists each result here, its connection's sole owner. Under ``--timings``
    each unit's record starts from parse and gains caption/summarize downstream.
    """
    units: list[DocUnit] = [
        DocUnit(
            item.document_id, item.file_name, item.file_hash,
            stages={} if timings else None,
        )
        for item in to_resume
    ]
    parse_phase = progress.phase("parse")
    parse_phase.start(len(to_parse))

    # Capped files (failed MAX_INGEST_ATTEMPTS× already) are skipped, not retried
    # — parse is the most expensive stage, so the gate precedes the pool, mirroring
    # the identical gates in caption.py/summary.py.
    to_parse_live: list[parsers.ParseRequest] = []
    for req in to_parse:
        if writer.is_capped(req.file_hash, "parse"):
            console.warn(
                f"{req.file_name}: skipping parse — failed "
                f"{MAX_INGEST_ATTEMPTS}× already; not retrying."
            )
            parse_phase.advance()
        else:
            to_parse_live.append(req)

    for outcome in pool.parse_stream(
        to_parse_live,
        parse_fn=parsers._parse_request,
        config=parse_config,
        max_workers=max_workers,
        warmup=parsers._warm_worker,
        verbose=verbose,
        required_models=required_models,
        on_progress=lambda ev: parse_phase.lane(ev.worker, ev.item, ev.stage),
    ):
        req = outcome.request
        # Notices the worker collected (it has no Live console of its own); emit
        # them here, where console.warn coordinates with the progress bar
        # instead of stomping it.
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

        persist_t = time.perf_counter()
        document_id = writer.persist_parse(outcome.parsed)
        stages = None
        if timings and outcome.parse_stages is not None:
            # Chunk INSERTs fold into `embed`, per #162 — they land here, just
            # after the worker's embed step, under the same bucket.
            stages = dict(outcome.parse_stages)
            stages["embed"] = (
                stages.get("embed", 0.0) + (time.perf_counter() - persist_t)
            )
        units.append(DocUnit(
            document_id, req.file_name, req.file_hash,
            page_count=outcome.parsed.page_count, stages=stages,
        ))
        parse_phase.advance()
    return units
