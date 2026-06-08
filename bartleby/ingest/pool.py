"""Parallel parse pool for ``bartleby scribe`` (#165).

Ingest's expensive step is *parsing* — cracking a PDF, OCR'ing sparse pages,
embedding chunks — and parsing one document never needs another, so it
parallelizes cleanly. This module runs the parse step across a pool of worker
*processes*; everything downstream (persist, caption, summarize) stays on the
main process behind the single :class:`~bartleby.ingest.writer.Writer`, which
remains the connection's sole owner on a single thread.

The pool is deliberately generic: it knows how to *schedule* a parse function
over a stream of requests, not how to parse. The caller supplies a picklable
``parse_fn(request, config) -> outcome`` and a picklable ``config``; the pool
ships those to each worker once and streams results back in completion order.
That keeps this module free of any dependency on ``commands.scribe`` (which owns
the real parse logic) — the function travels as data.

One seam, two schedulers: at ``max_workers <= 1`` the requests run inline in the
calling process (no process is spawned — the honest degenerate case, and the
path the mocked test suite drives); above that, a ``spawn`` pool runs them
concurrently. Both call the *same* ``parse_fn`` and yield the *same* outcomes,
so there is one parse pipeline, not two — only the scheduler differs.

``spawn`` (not ``fork``) is mandatory: the parse libraries load native, threaded
ML runtimes (docling's layout models, the embedding model's tokenizer pool) that
are unsafe to inherit across a fork on macOS.
"""

from __future__ import annotations

import multiprocessing
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, TypeVar

from bartleby.lib.consts import WORKER_MAX_TASKS

Req = TypeVar("Req")
Cfg = TypeVar("Cfg")
Out = TypeVar("Out")


@dataclass(frozen=True)
class ProgressEvent:
    """A worker → main-process progress beat for one in-flight parse.

    Carries only the tiny tuple the renderer needs — which worker, what file,
    which stage — never the parse payload. This is the cheap thing #165 said
    couldn't cross the queue: a few of these per file, not a page-level stream.
    """
    worker: str
    item: str
    stage: str


# Worker-process globals: the initializer stashes the parse function, its config,
# and the progress queue here once, so each task ships only its (small) request
# over the queue — not the function and run-wide config repeated per item.
_PARSE_FN: Callable | None = None
_CONFIG: object = None
_PROGRESS_Q: object = None


def _init_worker(parse_fn, config, warmup, verbose, required_models, progress_q) -> None:
    """Set up one spawned worker: stash the parse fn + config, quieten, warm.

    A spawned worker is a fresh interpreter. Environment set in the parent is
    inherited, but in-process warnings/logging config is not — so re-run the
    third-party quieting and offline gate here, then optionally pay the model
    load cost up front so the worker's first document doesn't eat it.
    """
    global _PARSE_FN, _CONFIG, _PROGRESS_Q
    _PARSE_FN = parse_fn
    _CONFIG = config
    _PROGRESS_Q = progress_q

    import sys

    from loguru import logger

    from bartleby.lib.quiet import setup_quiet_third_party

    setup_quiet_third_party(verbose=verbose, required_models=required_models)
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    if warmup is not None:
        warmup(config)


def _worker_report(request) -> Callable[[str], None]:
    """A ``report(stage)`` for one task that posts ``ProgressEvent``s home.

    The worker identity (its process name) and the file are constant across a
    task, so ``parse_fn`` only names the stage; we attach the rest. A noop when
    no progress queue is wired, and swallows queue errors so a progress hiccup
    can never break a parse.
    """
    q = _PROGRESS_Q
    if q is None:
        return lambda stage: None
    worker = multiprocessing.current_process().name
    item = getattr(request, "file_name", str(request))

    def report(stage: str) -> None:
        try:
            q.put(ProgressEvent(worker, item, stage))
        except Exception:
            pass

    return report


def _run_task(request):
    return _PARSE_FN(request, _CONFIG, _worker_report(request))


def _inline_report(
    request, on_progress: Callable[[ProgressEvent], None] | None
) -> Callable[[str], None]:
    """The ``report(stage)`` for the inline path — calls ``on_progress`` directly
    (no queue, same process), tagging one synthetic worker so the lane is stable."""
    if on_progress is None:
        return lambda stage: None
    item = getattr(request, "file_name", str(request))
    return lambda stage: on_progress(ProgressEvent("worker-1", item, stage))


def _drain_progress(queue, on_progress: Callable[[ProgressEvent], None]) -> None:
    """Pump worker progress events to ``on_progress`` until the ``None`` sentinel.

    Runs on its own daemon thread in the main process so it can render while the
    main thread blocks in ``imap_unordered``. ``None`` (never a real event) is the
    shutdown signal the stream posts once the pool is drained.
    """
    while True:
        event = queue.get()
        if event is None:
            return
        on_progress(event)


def parse_stream(
    requests: Iterable[Req],
    *,
    parse_fn: Callable[[Req, Cfg, Callable[[str], None]], Out],
    config: Cfg,
    max_workers: int,
    warmup: Callable[[Cfg], None] | None = None,
    verbose: bool = False,
    required_models: Iterable[str] = (),
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> Iterator[Out]:
    """Parse every request, yielding each outcome as it completes (unordered).

    At ``max_workers <= 1`` the requests run inline in this process; above that,
    across a ``spawn`` pool of that many workers. Both paths call
    ``parse_fn(request, config, report)`` and yield its return value, so the
    caller's drain loop is identical either way. ``report(stage)`` is how a parse
    announces its current stage; the pool tags it with the worker + file and, when
    ``on_progress`` is set, delivers a :class:`ProgressEvent` to the main process.

    ``parse_fn`` must not raise — it is expected to capture parse failures as
    data in its returned outcome, so one bad file never tears down the stream.
    ``warmup(config)`` runs once per worker *instance* in the pool path — i.e.
    again each time a worker is recycled after ``WORKER_MAX_TASKS`` docs (#213) —
    never inline, where the caller's process is already warm.
    """
    if max_workers <= 1:
        for request in requests:
            yield parse_fn(request, config, _inline_report(request, on_progress))
        return

    ctx = multiprocessing.get_context("spawn")
    # A Manager queue (proxy-based) pickles cleanly into spawn workers via the
    # initializer, where a bare ctx.Queue() does not. Progress is low-volume, so
    # the proxy overhead is noise. None when the caller wants no progress.
    manager = ctx.Manager() if on_progress is not None else None
    progress_q = manager.Queue() if manager is not None else None
    drain: threading.Thread | None = None
    if progress_q is not None:
        drain = threading.Thread(
            target=_drain_progress, args=(progress_q, on_progress), daemon=True,
        )
        drain.start()

    try:
        with ctx.Pool(
            processes=max_workers,
            # Recycle each worker after WORKER_MAX_TASKS docs so docling/torch RSS
            # can't grow unbounded across a long run (#213) — a replacement worker
            # re-runs _init_worker (re-warming the models).
            maxtasksperchild=WORKER_MAX_TASKS,
            initializer=_init_worker,
            initargs=(
                parse_fn, config, warmup, verbose, tuple(required_models), progress_q,
            ),
        ) as pool:
            # chunksize=1 (imap_unordered default): a worker pulls the next single
            # document whenever it's free, so a mixed corpus of cheap and expensive
            # files load-balances instead of being dealt out in fixed blocks.
            yield from pool.imap_unordered(_run_task, requests)
    finally:
        if progress_q is not None:
            progress_q.put(None)        # stop the drain thread
            drain.join(timeout=2)
            manager.shutdown()
