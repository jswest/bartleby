"""Per-stage wall-clock timing for ingest (issue #162).

Opt-in instrumentation. The scribe loop builds one :class:`StageTimer` per
document when ``--timings`` is set, driving it off the ``on_stage`` transitions
the pipeline already emits, then aggregates the per-doc records into docs/sec,
pages/sec, and a per-stage breakdown. Off by default — no ``StageTimer`` is
constructed and the ingest path is byte-for-byte unchanged.

The raw ``on_stage`` labels are mapped to a small canonical vocabulary so the
breakdown reads as parse / embed / caption / summarize regardless of which
converter ran. Time before the first stage mark (file hash + archive copy)
lands in ``prep``. The chunk ``INSERT``s are folded into ``embed`` — they
immediately follow ``embed_texts`` under the same label, and the baseline
question this exists to answer (is per-doc time dominated by parse or by the
~55 captions/doc?) doesn't need them split out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Raw on_stage labels (scribe.py) → canonical timing buckets. Unmapped labels
# pass through unchanged so a new stage shows up rather than vanishing.
_CANONICAL = {
    "extracting": "parse",
    "embedding": "embed",
    "analyzing images": "caption",
    "analyzing image": "caption",
    "summarizing": "summarize",
}

# Time before the first mark — file hash + archive copy.
PREP = "prep"

# Display order for the breakdown: prep, then pipeline order. Stages not listed
# here (an unmapped future label) are appended after, so nothing is dropped.
STAGE_ORDER = (PREP, "parse", "embed", "caption", "summarize")


def canonical_stage(label: str) -> str:
    return _CANONICAL.get(label, label)


def ordered_stages(stages: dict[str, float]) -> list[str]:
    """Stage keys in display order: STAGE_ORDER first, then any unmapped label
    (a future stage) appended so nothing is silently dropped."""
    known = [s for s in STAGE_ORDER if s in stages]
    return known + [s for s in stages if s not in STAGE_ORDER]


class StageTimer:
    """Accumulates wall-clock per canonical stage for one document.

    Construct at the top of a document's processing; call :meth:`mark` at each
    ``on_stage`` transition and :meth:`finish` once it's done. ``totals`` then
    holds ``{canonical_stage: seconds}`` and ``total`` the whole-document
    wall-clock (prep included).
    """

    def __init__(self) -> None:
        now = time.perf_counter()
        self._start = now
        self._last = now
        self._current = PREP
        self.totals: dict[str, float] = {}
        self.total: float = 0.0

    def _accumulate(self, now: float) -> None:
        self.totals[self._current] = (
            self.totals.get(self._current, 0.0) + (now - self._last)
        )
        self._last = now

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self._accumulate(now)
        self._current = canonical_stage(label)

    def finish(self) -> None:
        now = time.perf_counter()
        self._accumulate(now)
        self.total = now - self._start


@dataclass
class DocTiming:
    """One document's timing record, captured after a successful ingest.

    Only what the aggregate consumes: the page count (for pages/sec) and the
    per-stage seconds. The per-doc total and filename are emitted live to
    stderr in the scribe loop, not re-read from here.
    """

    page_count: int | None
    stages: dict[str, float] = field(default_factory=dict)


def render_doc_line(file_name: str, total_s: float, stages: dict[str, float]) -> str:
    """One per-document stderr line: total plus the per-stage split."""
    split = " · ".join(f"{s} {stages[s]:.2f}" for s in ordered_stages(stages))
    return f"{file_name}: {total_s:.2f}s ({split})"


def aggregate(records: list[DocTiming], wall_clock_s: float) -> dict:
    """Roll per-doc records into a benchmark summary.

    Rates use ``wall_clock_s`` — the whole-run wall-clock measured around the
    ingest loop — not the sum of per-doc totals, so the overhead between docs is
    counted and "pages/sec at N workers" stays comparable across runs. Per-stage
    ``pct`` is the share of summed stage time (the parse-vs-caption split);
    ``mean_s`` is the per-document mean.
    """
    docs = len(records)
    pages = sum(r.page_count or 0 for r in records)

    stage_totals: dict[str, float] = {}
    for record in records:
        for stage, secs in record.stages.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + secs
    summed = sum(stage_totals.values())

    stages = {
        stage: {
            "total_s": round(stage_totals[stage], 3),
            "pct": round(100 * stage_totals[stage] / summed, 1) if summed else 0.0,
            "mean_s": round(stage_totals[stage] / docs, 3) if docs else 0.0,
        }
        for stage in ordered_stages(stage_totals)
    }

    return {
        "docs": docs,
        "pages": pages,
        "wall_clock_s": round(wall_clock_s, 3),
        "docs_per_s": round(docs / wall_clock_s, 4) if wall_clock_s else 0.0,
        "pages_per_s": round(pages / wall_clock_s, 4) if wall_clock_s else 0.0,
        "stages": stages,
    }


def render_summary(agg: dict) -> list[str]:
    """Human-readable lines for the aggregate (stderr); the JSON is for capture."""
    lines = [
        f"{agg['docs']} docs · {agg['pages']} pages · "
        f"{agg['wall_clock_s']}s wall",
        f"{agg['docs_per_s']} docs/sec · {agg['pages_per_s']} pages/sec",
    ]
    for stage, vals in agg["stages"].items():
        lines.append(
            f"  {stage:<10} {vals['total_s']:>9.3f}s  "
            f"{vals['pct']:>5.1f}%  (mean {vals['mean_s']:.3f}s/doc)"
        )
    return lines
