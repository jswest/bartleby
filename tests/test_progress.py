"""The scribe progress renderer's state machine (issue #170).

These drive ScribeProgress's phase API directly without entering the Live
context — the tally math, run-of-show header, and lane bookkeeping are all
observable from the object's state, so no terminal or render loop is needed.
"""

from __future__ import annotations

import sys

from rich.console import Console

import bartleby.lib.console as console_mod
from bartleby.ingest.progress import ScribeProgress, _RESERVED_ROWS


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


def test_lane_overflow_is_dropped_not_fatal():
    # More live workers than lanes can't happen (lanes ≥ widest phase), but if it
    # did the extra worker is silently dropped rather than crashing the render.
    sp = ScribeProgress(n_lanes=1)
    par = sp.phase("parse")
    par.start(3)
    par.lane("w1", "a", "x")
    par.lane("w2", "b", "x")
    assert set(sp._by_key) == {"w1"}
