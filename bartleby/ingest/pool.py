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
from typing import Callable, Iterable, Iterator, TypeVar

Req = TypeVar("Req")
Cfg = TypeVar("Cfg")
Out = TypeVar("Out")


# Worker-process globals: the initializer stashes the parse function and its
# config here once, so each task ships only its (small) request over the queue —
# not the function and run-wide config repeated per item.
_PARSE_FN: Callable | None = None
_CONFIG: object = None


def _init_worker(parse_fn, config, warmup, verbose, required_models) -> None:
    """Set up one spawned worker: stash the parse fn + config, quieten, warm.

    A spawned worker is a fresh interpreter. Environment set in the parent is
    inherited, but in-process warnings/logging config is not — so re-run the
    third-party quieting and offline gate here, then optionally pay the model
    load cost up front so the worker's first document doesn't eat it.
    """
    global _PARSE_FN, _CONFIG
    _PARSE_FN = parse_fn
    _CONFIG = config

    import sys

    from loguru import logger

    from bartleby.lib.quiet import setup_quiet_third_party

    setup_quiet_third_party(verbose=verbose, required_models=required_models)
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    if warmup is not None:
        warmup(config)


def _run_task(request):
    return _PARSE_FN(request, _CONFIG)


def parse_stream(
    requests: Iterable[Req],
    *,
    parse_fn: Callable[[Req, Cfg], Out],
    config: Cfg,
    max_workers: int,
    warmup: Callable[[Cfg], None] | None = None,
    verbose: bool = False,
    required_models: Iterable[str] = (),
) -> Iterator[Out]:
    """Parse every request, yielding each outcome as it completes (unordered).

    At ``max_workers <= 1`` the requests run inline in this process; above that,
    across a ``spawn`` pool of that many workers. Both paths call
    ``parse_fn(request, config)`` and yield its return value, so the caller's
    drain loop is identical either way.

    ``parse_fn`` must not raise — it is expected to capture parse failures as
    data in its returned outcome, so one bad file never tears down the stream.
    ``warmup(config)`` runs once per worker in the pool path (never inline,
    where the caller's process is already warm).
    """
    if max_workers <= 1:
        for request in requests:
            yield parse_fn(request, config)
        return

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(parse_fn, config, warmup, verbose, tuple(required_models)),
    ) as pool:
        # chunksize=1 (imap_unordered default): a worker pulls the next single
        # document whenever it's free, so a mixed corpus of cheap and expensive
        # files load-balances instead of being dealt out in fixed blocks.
        yield from pool.imap_unordered(_run_task, requests)
