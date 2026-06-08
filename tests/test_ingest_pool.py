"""The generic parse-pool executor (#165).

These tests drive `pool.parse_stream` with a trivial, model-free `parse_fn`
defined at module scope — so it pickles across the `spawn` boundary the same way
`scribe._parse_request` does, but without loading any ML models. That isolates
the *scheduling* contract (inline vs. pooled, all results returned, order
independence, config + warmup delivery) from the parse logic itself, which the
end-to-end `test_scribe.py` suite covers on the inline path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from bartleby.ingest import pool


@dataclass
class _Req:
    n: int

    @property
    def file_name(self) -> str:
        # Mirrors ParseRequest.file_name, which the pool tags progress events with.
        return f"doc-{self.n}"


# Module-level so spawn can re-import them by qualified name in each worker.
def _square(request: _Req, config: dict, report) -> dict:
    report("squaring")
    return {"n": request.n, "sq": request.n * request.n, "bias": config["bias"]}


def _pid(request: _Req, config: dict, report) -> dict:
    # Reports the serving worker's PID so a test can see workers get recycled.
    return {"n": request.n, "pid": os.getpid()}


def _boom_on_three(request: _Req, config: dict, report) -> dict | None:
    # A parse_fn is expected to capture its own failures as data; here we model a
    # fn that returns a sentinel instead of raising, so the stream never breaks.
    if request.n == 3:
        return {"n": 3, "error": "boom"}
    return {"n": request.n, "sq": request.n * request.n, "bias": config["bias"]}


def test_parse_stream_inline_runs_in_process():
    out = list(pool.parse_stream(
        [_Req(1), _Req(2), _Req(3)],
        parse_fn=_square, config={"bias": 10}, max_workers=1,
    ))
    assert sorted(o["sq"] for o in out) == [1, 4, 9]
    assert all(o["bias"] == 10 for o in out)


def test_parse_stream_inline_empty_yields_nothing():
    assert list(pool.parse_stream(
        [], parse_fn=_square, config={"bias": 0}, max_workers=1,
    )) == []


def test_parse_stream_pool_spawns_and_returns_all():
    requests = [_Req(i) for i in range(8)]
    out = list(pool.parse_stream(
        requests, parse_fn=_square, config={"bias": 10}, max_workers=3,
    ))
    # Every request comes back exactly once, regardless of completion order, and
    # carries the run-wide config the initializer shipped to each worker.
    assert sorted(o["n"] for o in out) == list(range(8))
    assert sorted(o["sq"] for o in out) == [i * i for i in range(8)]
    assert all(o["bias"] == 10 for o in out)


def test_parse_stream_pool_recycles_workers(monkeypatch):
    # maxtasksperchild recycles a worker after WORKER_MAX_TASKS docs so a long run
    # can't grow RSS unbounded (#213). With a cap of 2 over 8 tasks, more than
    # max_workers distinct processes must have served them — and every task still
    # comes back exactly once across the recycles.
    monkeypatch.setattr(pool, "WORKER_MAX_TASKS", 2)
    requests = [_Req(i) for i in range(8)]
    out = list(pool.parse_stream(
        requests, parse_fn=_pid, config={}, max_workers=2,
    ))
    assert sorted(o["n"] for o in out) == list(range(8))
    assert len({o["pid"] for o in out}) > 2


def test_parse_stream_pool_does_not_break_on_one_bad_item():
    requests = [_Req(i) for i in range(5)]
    out = list(pool.parse_stream(
        requests, parse_fn=_boom_on_three, config={"bias": 1}, max_workers=2,
    ))
    assert sorted(o["n"] for o in out) == [0, 1, 2, 3, 4]      # all 5 returned
    bad = [o for o in out if o.get("error")]
    assert len(bad) == 1 and bad[0]["n"] == 3                  # the failure rode back as data


def test_parse_stream_pool_runs_warmup_once_per_worker():
    # warmup runs in each worker's initializer; its effect can't cross back, so
    # we just assert the run completes with warmup wired (a warmup that raised
    # would tear the pool down and fail the run).
    out = list(pool.parse_stream(
        [_Req(1), _Req(2)],
        parse_fn=_square, config={"bias": 0}, max_workers=2,
        warmup=_noop_warmup,
    ))
    assert sorted(o["n"] for o in out) == [1, 2]


def _noop_warmup(config: dict) -> None:
    return None


def test_parse_stream_inline_reports_progress():
    # The inline path delivers ProgressEvents straight to on_progress (same
    # process, no queue), tagged with a stable synthetic worker + the file name.
    events: list[pool.ProgressEvent] = []
    out = list(pool.parse_stream(
        [_Req(1), _Req(2)],
        parse_fn=_square, config={"bias": 0}, max_workers=1,
        on_progress=events.append,
    ))
    assert sorted(o["n"] for o in out) == [1, 2]
    assert {(e.worker, e.item, e.stage) for e in events} == {
        ("worker-1", "doc-1", "squaring"),
        ("worker-1", "doc-2", "squaring"),
    }


def test_parse_stream_inline_no_progress_callback_is_noop():
    # report() is a noop when no on_progress is wired — parse_fn still calls it.
    out = list(pool.parse_stream(
        [_Req(1)], parse_fn=_square, config={"bias": 0}, max_workers=1,
    ))
    assert out[0]["sq"] == 1


def test_parse_stream_pool_reports_progress_per_worker():
    # Across the spawn pool, each worker posts its beats home over the queue; the
    # drain thread (joined before parse_stream returns) delivers them all. Every
    # file is reported, and the worker tags are real pool-process names.
    events: list[pool.ProgressEvent] = []
    requests = [_Req(i) for i in range(6)]
    out = list(pool.parse_stream(
        requests, parse_fn=_square, config={"bias": 0}, max_workers=2,
        on_progress=events.append,
    ))
    assert sorted(o["n"] for o in out) == list(range(6))
    assert {e.item for e in events} == {f"doc-{i}" for i in range(6)}
    assert all(e.stage == "squaring" for e in events)
    assert all(e.worker.startswith("SpawnPoolWorker") for e in events)
