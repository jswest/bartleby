"""The scribe ingest progress display (issue #170).

Ingest runs as three sequential phases behind the single Writer — parse (a pool
of worker *processes*), then caption, then summarize (each a thread pool in this
process). This module renders that, scaled to N workers, as a single live view:

- a **run-of-show** header naming the three phases with a live ``done/total``
  tally each, so the whole pipeline reads at a glance and the active phase stands
  out (a phase whose total isn't known yet shows ``—``); the active phase also
  shows an estimated time-to-done (``~22m left``) once it has warmed up (#209);
- one **overall bar** counting *all* pipeline units — files parsed + images
  captioned + documents summarized — with its total revealed phase by phase, so
  it fills start to finish rather than pinning at 100% after parse;
- one **lane** per worker (``● <file> · <stage>``), showing what each parse
  process / caption thread / summarize thread is chewing on right now.

The whole thing is the sole renderer: parse workers can't draw (they're other
processes), so they post :class:`~bartleby.ingest.pool.ProgressEvent`s home and
this renders them; the in-process caption/summarize threads call the lane API
directly. Skip/error lines still print above the bars — they go through the
shared :mod:`bartleby.lib.console`, whose Rich console this display shares, so a
``console.warn`` inserts above the live region instead of stomping it.

Lanes are *sticky per worker*: a key (a parse process name, or a caption/
summarize thread id) claims a lane on first sight and keeps it for the phase, its
label updating as that worker moves to the next item. Phase boundaries clear the
lanes so caption rows don't show leftover parse files.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from bartleby.lib import console

# Pipeline phases, in run order — the run-of-show header and the overall total
# are both built straight off this tuple.
PHASES = ("parse", "caption", "summarize")

# Rows the live region needs beyond its lanes: the run-of-show header, the
# overall bar, and a line of breathing room. Lanes are capped to the terminal
# height minus this so the whole region always fits the screen (#208).
_RESERVED_ROWS = 4

# Per-phase ETA (#209). The estimate is a rate over a *trailing window* of the
# most recent (clock, done) samples, not a cumulative average — docling parse
# times vary wildly per doc, so a moving window keeps the estimate from lurching.
_ETA_SAMPLES = 64           # window length, in samples (one per advance)
_ETA_MIN_SAMPLES = 3        # cold-start guard: too few samples → no estimate yet
_ETA_MIN_ELAPSED = 1.0      # …and too short a span is just as unreliable (seconds)


def _fmt_eta(secs: float) -> str:
    """A compact, human duration for the header: ``45s`` / ``22m`` / ``1h03m``."""
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins:02d}m"


class ScribeProgress:
    """Live multi-worker progress for a scribe ingest run (see module docstring).

    Construct with the maximum number of workers any phase will use (lanes are
    reused across phases), enter as a context manager for the run, and drive each
    phase through :meth:`phase`. Thread-safe: the lane bookkeeping is guarded by a
    lock and Rich's own progress/live locking covers the rendering, so the parse
    drain thread and the caption/summarize worker threads can all report freely.
    """

    def __init__(
        self, *, n_lanes: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._lock = threading.RLock()
        self._total: dict[str, int] = {p: 0 for p in PHASES}
        self._done: dict[str, int] = {p: 0 for p in PHASES}
        self._known: dict[str, bool] = {p: False for p in PHASES}
        self._active: str | None = None

        # Per-phase ETA (#209): a trailing window of (clock, done) samples for the
        # active phase, and the remaining-seconds estimate derived from it. Only
        # the active phase ever carries an estimate, so one slot suffices. The
        # clock is injectable so tests can advance time deterministically.
        self._clock = clock
        self._samples: deque[tuple[float, int]] = deque(maxlen=_ETA_SAMPLES)
        self._active_eta: float | None = None

        shared = console.get_console()

        # Cap lanes so the live region (header + overall bar + one row per lane)
        # can't outgrow the terminal. Rich's Live only redraws in place while its
        # renderable fits the screen; once it's taller, the top scrolls out of
        # reach and every refresh stacks a fresh frame below the last instead of
        # overwriting it (#208). Only a real TTY redraws in place — off a terminal
        # (pipe, CI log, tests) Live appends regardless, so leave the count alone.
        if shared.is_terminal:
            n_lanes = min(n_lanes, shared.size.height - _RESERVED_ROWS)
        n_lanes = max(1, n_lanes)

        self._overall = Progress(
            TextColumn("[bold]{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),       # whole-run ETA, computed by Rich (#209)
            console=shared,
        )
        self._overall_task = self._overall.add_task(
            "overall", total=0, label="Overall",
        )
        # Lanes get a spinner + free-text label only — no bar/count/elapsed, since
        # a lane is "what this worker is doing now", not a measured quantity.
        self._lanes = Progress(
            SpinnerColumn(),
            TextColumn("{task.fields[label]}"),
            console=shared,
        )
        self._lane_tasks = [
            self._lanes.add_task("lane", label="", visible=False)
            for _ in range(n_lanes)
        ]
        self._free: list[int] = list(range(len(self._lane_tasks)))
        # Insertion-ordered so the oldest-active key can be evicted (LRU) when a
        # new worker needs a lane and none are free — see _lane_update.
        self._by_key: "OrderedDict[object, int]" = OrderedDict()

        # crop (don't stack) if the region still overshoots — e.g. the window is
        # resized shorter mid-run. Header + overall sit atop the group, so crop
        # sheds excess lanes from the bottom first (#208).
        self._live = Live(
            self, console=shared, refresh_per_second=8, vertical_overflow="crop"
        )

    # -- rendering -----------------------------------------------------------

    def __rich__(self) -> Group:
        return Group(self._header(), self._overall, self._lanes)

    def _header(self) -> Text:
        segments: list[Text] = []
        for phase in PHASES:
            tally = (
                f"{self._done[phase]}/{self._total[phase]}"
                if self._known[phase] else "—"
            )
            label = f"{phase} {tally}"
            # Only the active phase shows an ETA, and only once it's warmed up;
            # a finished phase has cleared it, a pending one never had one.
            if phase == self._active and self._active_eta is not None:
                label += f" · ~{_fmt_eta(self._active_eta)} left"
            seg = Text(label)
            if phase == self._active:
                seg.stylize("bold cyan")
            elif self._known[phase] and self._done[phase] >= self._total[phase]:
                seg.stylize("dim green")          # finished
            else:
                seg.stylize("dim")                # pending / unknown
            segments.append(seg)
        return Text("Run of show  ", style="bold") + Text(" · ").join(segments)

    def _refresh_overall(self) -> None:
        # Total counts only phases whose denominator is known yet — so the bar
        # grows as parse hands off to caption and caption to summarize.
        total = sum(self._total[p] for p in PHASES if self._known[p])
        done = sum(self._done[p] for p in PHASES)
        self._overall.update(self._overall_task, total=total, completed=done)

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "ScribeProgress":
        self._live.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._clear_lanes()       # leave a clean header + filled overall bar
        self._live.stop()
        return False

    # -- lanes ---------------------------------------------------------------

    def _clear_lanes(self) -> None:
        with self._lock:
            for task in self._lane_tasks:
                self._lanes.update(task, label="", visible=False)
            self._free = list(range(len(self._lane_tasks)))
            self._by_key.clear()

    def _lane_update(self, key: object, item: str, stage: str) -> None:
        with self._lock:
            idx = self._by_key.get(key)
            if idx is None:
                if not self._free:
                    # No free lane: a recycled parse worker (#213) reports under a
                    # new process name while a dead worker still holds a lane. Evict
                    # the least-recently-updated key — the dead worker, which has
                    # stopped reporting — and reuse its lane for this live one.
                    _, idx = self._by_key.popitem(last=False)
                else:
                    idx = self._free.pop(0)
                self._by_key[key] = idx
            else:
                self._by_key.move_to_end(key)   # mark recently-active for LRU
            label = f"{console.truncate_filename(item)} · {stage}"
        # Updated outside the lock (Rich has its own); safe because each key is a
        # single worker reporting serially, so two updates for one lane can't race.
        self._lanes.update(self._lane_tasks[idx], label=label, visible=True)

    # -- phases --------------------------------------------------------------

    def phase(self, name: str) -> "_Phase":
        """A handle for driving ``name``'s tally and lanes (see :class:`_Phase`)."""
        return _Phase(self, name)

    def _recompute_eta(self, name: str) -> None:
        """Refresh the active phase's remaining-time estimate from a trailing
        window of recent ``(clock, done)`` samples (caller holds the lock).

        Computed eagerly here — inside the lock that guards ``_samples`` — and
        stashed in ``_active_eta`` so :meth:`_header` reads one plain float on the
        render thread instead of racing the deque."""
        done, total = self._done[name], self._total[name]
        self._samples.append((self._clock(), done))
        if done >= total or len(self._samples) < _ETA_MIN_SAMPLES:
            self._active_eta = None       # finished, or too little to go on yet
            return
        (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
        span, progressed = t1 - t0, d1 - d0
        if span < _ETA_MIN_ELAPSED or progressed <= 0:
            self._active_eta = None       # stalled or barely started — no guess
            return
        self._active_eta = (total - done) * span / progressed

    def _start(self, name: str, total: int) -> None:
        with self._lock:
            self._active = name
            self._total[name] = total
            self._done[name] = 0
            self._known[name] = True
            self._samples.clear()         # a phase's rate is its own, not the last's
            self._recompute_eta(name)
        self._clear_lanes()       # new phase, fresh lanes
        self._refresh_overall()

    def _advance(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._done[name] += n
            self._recompute_eta(name)
        self._refresh_overall()


class _Phase:
    """A bound view of one phase, handed to the phase's loop so it never touches
    Rich directly. ``start`` reveals the denominator (and grows the overall bar),
    ``advance`` moves the tally, ``lane`` updates a worker row."""

    def __init__(self, parent: ScribeProgress, name: str) -> None:
        self._parent = parent
        self._name = name

    def start(self, total: int) -> None:
        self._parent._start(self._name, total)

    def advance(self, n: int = 1) -> None:
        self._parent._advance(self._name, n)

    def lane(self, key: object, item: str, stage: str) -> None:
        self._parent._lane_update(key, item, stage)
