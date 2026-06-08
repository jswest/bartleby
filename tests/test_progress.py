"""The scribe progress renderer's state machine (issue #170).

These drive ScribeProgress's phase API directly without entering the Live
context — the tally math, run-of-show header, and lane bookkeeping are all
observable from the object's state, so no terminal or render loop is needed.
"""

from __future__ import annotations

from bartleby.ingest.progress import ScribeProgress


def _overall(sp: ScribeProgress) -> tuple[float, float]:
    task = sp._overall.tasks[0]
    return task.completed, task.total


def test_overall_total_grows_as_each_phase_reveals_its_denominator():
    # The overall bar counts every pipeline unit, but a phase's units only join
    # the total once that phase starts — so the bar fills start to finish rather
    # than pinning at 100% after parse.
    sp = ScribeProgress(n_lanes=4)

    par = sp.phase("parse")
    par.start(3)
    assert _overall(sp) == (0, 3)
    par.advance()
    par.advance()
    assert _overall(sp) == (2, 3)              # 2 parsed, caption/summarize unknown

    cap = sp.phase("caption")
    cap.start(5)
    assert _overall(sp) == (2, 8)              # caption's 5 join the total
    cap.set_completed(5)
    assert _overall(sp) == (7, 8)

    summ = sp.phase("summarize")
    summ.start(2)
    assert _overall(sp) == (7, 10)
    summ.advance()
    summ.advance()
    assert _overall(sp) == (9, 10)             # everything done


def test_header_shows_dash_until_known_then_the_tally():
    sp = ScribeProgress(n_lanes=2)
    # Nothing started — every phase reads as not-yet-known.
    assert "parse —" in sp._header().plain
    assert "caption —" in sp._header().plain

    sp.phase("parse").start(4)
    header = sp._header().plain
    assert "parse 0/4" in header
    assert "caption —" in header and "summarize —" in header


def test_lanes_are_sticky_per_key_and_reset_between_phases():
    sp = ScribeProgress(n_lanes=2)
    par = sp.phase("parse")
    par.start(2)
    par.lane("w1", "a.pdf", "extracting")
    par.lane("w2", "b.pdf", "extracting")
    par.lane("w1", "a.pdf", "embedding")       # same worker reuses its lane
    assert set(sp._by_key) == {"w1", "w2"}

    # A new phase frees every lane so caption rows don't show leftover parse files.
    sp.phase("caption").start(1)
    assert sp._by_key == {}


def test_lane_reclaim_evicts_lru_when_out_of_lanes():
    # No free lane → the least-recently-updated key is evicted and its lane reused
    # (the dead recycled worker, #213), so a live worker is never dropped.
    sp = ScribeProgress(n_lanes=1)
    par = sp.phase("parse")
    par.start(3)
    par.lane("w1", "a", "x")
    assert set(sp._by_key) == {"w1"}
    par.lane("w2", "b", "x")           # no free lane → evict w1 (LRU), reuse its lane
    assert set(sp._by_key) == {"w2"}


def test_lane_reclaim_keeps_the_most_recently_active_workers():
    # Eviction is least-recently-*updated*, not least-recently-added: touching a
    # worker keeps its lane, so the rows always track the currently-active workers.
    sp = ScribeProgress(n_lanes=2)
    par = sp.phase("parse")
    par.start(5)
    par.lane("w1", "a", "x")
    par.lane("w2", "b", "x")
    par.lane("w1", "a", "embedding")   # touch w1 → w2 is now the LRU
    par.lane("w3", "c", "x")           # full → evict w2, not w1
    assert set(sp._by_key) == {"w1", "w3"}
