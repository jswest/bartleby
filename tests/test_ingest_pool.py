"""The generic parse-pool executor (#165).

These tests drive `pool.parse_stream` with a trivial, model-free `parse_fn`
defined at module scope — so it pickles across the `spawn` boundary the same way
`scribe._parse_request` does, but without loading any ML models. That isolates
the *scheduling* contract (inline vs. pooled, all results returned, order
independence, config + warmup delivery) from the parse logic itself, which the
end-to-end `test_scribe.py` suite covers on the inline path.
"""

from __future__ import annotations

from dataclasses import dataclass

from bartleby.ingest import pool


@dataclass
class _Req:
    n: int


# Module-level so spawn can re-import them by qualified name in each worker.
def _square(request: _Req, config: dict) -> dict:
    return {"n": request.n, "sq": request.n * request.n, "bias": config["bias"]}


def _boom_on_three(request: _Req, config: dict) -> dict | None:
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
