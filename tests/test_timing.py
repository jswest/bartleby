"""Unit tests for the opt-in ingest timing helpers (issue #162)."""

from __future__ import annotations

import pytest

from bartleby.lib import timing
from bartleby.lib.timing import DocTiming, StageTimer, aggregate, canonical_stage


def _clock(monkeypatch, ticks):
    """Drive timing.StageTimer off a deterministic perf_counter sequence."""
    it = iter(ticks)
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(it))


def test_canonical_stage_maps_raw_labels():
    assert canonical_stage("extracting") == "parse"
    assert canonical_stage("embedding") == "embed"
    assert canonical_stage("analyzing images") == "caption"
    assert canonical_stage("analyzing image") == "caption"
    assert canonical_stage("summarizing") == "summarize"


def test_canonical_stage_passes_unknown_through():
    # An unmapped future label shows up rather than vanishing.
    assert canonical_stage("rasterizing") == "rasterizing"


def test_stage_timer_accumulates_between_marks(monkeypatch):
    # init@0, mark parse@1, mark embed@3, finish@6.
    _clock(monkeypatch, [0, 1, 3, 6])
    timer = StageTimer()
    timer.mark("extracting")
    timer.mark("embedding")
    timer.finish()

    assert timer.totals == {"prep": 1, "parse": 2, "embed": 3}
    assert timer.total == 6


def test_stage_timer_sums_repeated_stage(monkeypatch):
    # caption is entered twice and its two intervals add up.
    # init@0, parse@2, caption@5 (→6: 1s), embed@6, caption@9 (→10: 1s), finish@10.
    _clock(monkeypatch, [0, 2, 5, 6, 9, 10])
    timer = StageTimer()
    timer.mark("extracting")
    timer.mark("analyzing images")
    timer.mark("embedding")
    timer.mark("analyzing image")
    timer.finish()

    assert timer.totals["caption"] == pytest.approx(1 + 1)
    assert timer.totals["parse"] == pytest.approx(3)
    assert timer.total == 10


def test_aggregate_rolls_up_rates_and_stage_split():
    records = [
        DocTiming(page_count=2, stages={"parse": 2, "embed": 4}),
        DocTiming(page_count=3, stages={"parse": 1, "caption": 3}),
    ]
    agg = aggregate(records, wall_clock_s=10.0)

    assert agg["docs"] == 2
    assert agg["pages"] == 5
    assert agg["wall_clock_s"] == 10.0
    assert agg["docs_per_s"] == 0.2
    assert agg["pages_per_s"] == 0.5

    # Pipeline order: parse, embed, caption. summed stage time = 10s.
    assert list(agg["stages"]) == ["parse", "embed", "caption"]
    assert agg["stages"]["parse"] == {"total_s": 3.0, "pct": 30.0, "mean_s": 1.5}
    assert agg["stages"]["embed"] == {"total_s": 4.0, "pct": 40.0, "mean_s": 2.0}
    assert agg["stages"]["caption"] == {"total_s": 3.0, "pct": 30.0, "mean_s": 1.5}


def test_aggregate_appends_unmapped_stage_after_known_order():
    records = [
        DocTiming(page_count=1, stages={"rasterizing": 1, "parse": 2}),
    ]
    agg = aggregate(records, wall_clock_s=3.0)
    # Known stage first (STAGE_ORDER), unmapped one trailing.
    assert list(agg["stages"]) == ["parse", "rasterizing"]


def test_aggregate_handles_missing_page_counts():
    records = [
        DocTiming(page_count=None, stages={"parse": 1}),
    ]
    agg = aggregate(records, wall_clock_s=2.0)
    assert agg["pages"] == 0
    assert agg["pages_per_s"] == 0.0


def test_aggregate_empty_is_safe():
    agg = aggregate([], wall_clock_s=0.0)
    assert agg["docs"] == 0
    assert agg["pages"] == 0
    assert agg["docs_per_s"] == 0.0
    assert agg["pages_per_s"] == 0.0
    assert agg["stages"] == {}


def test_render_doc_line_shows_total_and_ordered_split():
    line = timing.render_doc_line(
        "a.pdf", 6.0, {"embed": 4.0, "parse": 2.0},
    )
    # Total first, then the split in STAGE_ORDER (parse before embed).
    assert line == "a.pdf: 6.00s (parse 2.00 · embed 4.00)"


def test_render_summary_includes_rates_and_stage_lines():
    records = [
        DocTiming(page_count=2, stages={"parse": 2, "embed": 4}),
    ]
    lines = timing.render_summary(aggregate(records, wall_clock_s=6.0))
    blob = "\n".join(lines)
    assert "docs/sec" in blob
    assert "pages/sec" in blob
    assert "parse" in blob
    assert "embed" in blob
