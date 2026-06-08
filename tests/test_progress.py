"""The scribe progress renderer's state machine (issue #170).

These drive ScribeProgress's phase API directly without entering the Live
context — the tally math, run-of-show header, and lane bookkeeping are all
observable from the object's state, so no terminal or render loop is needed.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.progress import TimeRemainingColumn

import bartleby.lib.console as console_mod
from bartleby.ingest.progress import (
    ScribeProgress,
    _ETA_MIN_SAMPLES,
    _RESERVED_ROWS,
    _fmt_eta,
)


class _FakeClock:
    """A hand-cranked monotonic clock so ETA math is deterministic in tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> "_FakeClock":
        self.t += dt
        return self


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


def test_lanes_capped_to_terminal_height_on_a_tty(monkeypatch):
    # On a short terminal the lane region is bounded so header + overall + lanes
    # can't outgrow the screen and stack (#208). Force a 10-row TTY and ask for
    # more workers than fit.
    short_tty = Console(file=sys.stderr, force_terminal=True, height=10)
    monkeypatch.setattr(console_mod, "get_console", lambda: short_tty)

    sp = ScribeProgress(n_lanes=16)

    assert len(sp._lane_tasks) == 10 - _RESERVED_ROWS          # capped to fit
    # The whole region (lanes + the reserved header/overall rows) stays on screen.
    assert len(sp._lane_tasks) + _RESERVED_ROWS <= short_tty.size.height


def test_lanes_not_capped_off_a_terminal(monkeypatch):
    # Off a TTY (pipe, CI log, tests) Live appends rather than redrawing, so the
    # stacking bug can't occur and capping would only hide lanes — leave it alone.
    not_a_tty = Console(file=sys.stderr, height=10)  # is_terminal False
    monkeypatch.setattr(console_mod, "get_console", lambda: not_a_tty)

    sp = ScribeProgress(n_lanes=16)

    assert len(sp._lane_tasks) == 16


def test_live_region_crops_rather_than_stacking():
    # The Live is told to crop an over-tall region instead of stacking frames,
    # covering a mid-run resize the construction-time cap can't foresee (#208).
    sp = ScribeProgress(n_lanes=2)
    assert sp._live.vertical_overflow == "crop"


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


# -- per-phase ETA (#209) ----------------------------------------------------


def test_fmt_eta_is_a_compact_duration():
    assert _fmt_eta(45) == "45s"
    assert _fmt_eta(59.9) == "59s"          # truncates, never rounds up to a minute
    assert _fmt_eta(60) == "1m"
    assert _fmt_eta(22 * 60) == "22m"
    assert _fmt_eta(3600 + 3 * 60) == "1h03m"


def test_overall_bar_has_a_time_remaining_column():
    sp = ScribeProgress(n_lanes=2)
    assert any(isinstance(c, TimeRemainingColumn) for c in sp._overall.columns)


def test_eta_appears_on_the_active_phase_after_warmup():
    # Two even ticks → a rate of 1 item/sec, so 8 of 10 left reads as ~8s.
    clock = _FakeClock()
    sp = ScribeProgress(n_lanes=2, clock=clock)
    par = sp.phase("parse")
    par.start(10)
    clock.tick(1); par.advance()
    clock.tick(1); par.advance()

    header = sp._header().plain
    assert "parse 2/10 · ~8s left" in header
    assert header.count("left") == 1        # caption/summarize (pending) show none


def test_eta_withheld_until_enough_samples():
    clock = _FakeClock()
    sp = ScribeProgress(n_lanes=2, clock=clock)
    par = sp.phase("parse")
    par.start(10)
    # One sample short of the cold-start minimum → still no estimate.
    for _ in range(_ETA_MIN_SAMPLES - 2):
        clock.tick(1); par.advance()

    assert sp._active_eta is None
    assert "left" not in sp._header().plain


def test_eta_withheld_until_enough_time_has_elapsed():
    # Enough samples but a sub-second span is too little signal to trust.
    clock = _FakeClock()
    sp = ScribeProgress(n_lanes=2, clock=clock)
    par = sp.phase("parse")
    par.start(10)
    clock.tick(0.2); par.advance()
    clock.tick(0.2); par.advance()

    assert len(sp._samples) >= _ETA_MIN_SAMPLES
    assert sp._active_eta is None


def test_eta_clears_when_the_phase_finishes():
    clock = _FakeClock()
    sp = ScribeProgress(n_lanes=2, clock=clock)
    par = sp.phase("parse")
    par.start(3)
    clock.tick(1); par.advance()
    clock.tick(1); par.advance()
    assert sp._active_eta is not None       # warmed up mid-phase

    clock.tick(1); par.advance()            # 3/3 — done, no ETA to show
    assert sp._active_eta is None
    assert "left" not in sp._header().plain


def test_eta_does_not_bleed_across_a_phase_boundary():
    clock = _FakeClock()
    sp = ScribeProgress(n_lanes=2, clock=clock)
    par = sp.phase("parse")
    par.start(5)
    clock.tick(1); par.advance()
    clock.tick(1); par.advance()            # parse has an ETA

    clock.tick(1); sp.phase("caption").start(8)   # fresh phase resets the window
    assert sp._active == "caption"
    assert sp._active_eta is None
    assert "left" not in sp._header().plain
