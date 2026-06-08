"""The scribe ingest progress display (issue #170).

Ingest runs as three sequential phases behind the single Writer — parse (a pool
of worker *processes*), then caption, then summarize (each a thread pool in this
process). This module renders that, scaled to N workers, as a single live view:

- a **run-of-show** header naming the three phases with a live ``done/total``
  tally each, so the whole pipeline reads at a glance and the active phase stands
  out (a phase whose total isn't known yet shows ``—``);
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

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from bartleby.lib import console

# Pipeline phases, in run order — the run-of-show header and the overall total
# are both built straight off this tuple.
PHASES = ("parse", "caption", "summarize")


class ScribeProgress:
    """Live multi-worker progress for a scribe ingest run (see module docstring).

    Construct with the maximum number of workers any phase will use (lanes are
    reused across phases), enter as a context manager for the run, and drive each
    phase through :meth:`phase`. Thread-safe: the lane bookkeeping is guarded by a
    lock and Rich's own progress/live locking covers the rendering, so the parse
    drain thread and the caption/summarize worker threads can all report freely.
    """

    def __init__(self, *, n_lanes: int) -> None:
        self._lock = threading.RLock()
        self._total: dict[str, int] = {p: 0 for p in PHASES}
        self._done: dict[str, int] = {p: 0 for p in PHASES}
        self._known: dict[str, bool] = {p: False for p in PHASES}
        self._active: str | None = None

        shared = console.get_console()
        self._overall = Progress(
            TextColumn("[bold]{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
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
            for _ in range(max(1, n_lanes))
        ]
        self._free: list[int] = list(range(len(self._lane_tasks)))
        self._by_key: dict[object, int] = {}

        self._live = Live(self, console=shared, refresh_per_second=8)

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
            seg = Text(f"{phase} {tally}")
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
                    return        # more live workers than lanes — shouldn't happen
                idx = self._free.pop(0)
                self._by_key[key] = idx
            label = f"{console.truncate_filename(item)} · {stage}"
        # Updated outside the lock (Rich has its own); safe because each key is a
        # single worker reporting serially, so two updates for one lane can't race.
        self._lanes.update(self._lane_tasks[idx], label=label, visible=True)

    # -- phases --------------------------------------------------------------

    def phase(self, name: str) -> "_Phase":
        """A handle for driving ``name``'s tally and lanes (see :class:`_Phase`)."""
        return _Phase(self, name)

    def _start(self, name: str, total: int) -> None:
        with self._lock:
            self._active = name
            self._total[name] = total
            self._done[name] = 0
            self._known[name] = True
        self._clear_lanes()       # new phase, fresh lanes
        self._refresh_overall()

    def _set_completed(self, name: str, done: int) -> None:
        with self._lock:
            self._done[name] = done
        self._refresh_overall()

    def _advance(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._done[name] += n
        self._refresh_overall()


class _Phase:
    """A bound view of one phase, handed to the phase's loop so it never touches
    Rich directly. ``start`` reveals the denominator (and grows the overall bar),
    ``advance``/``set_completed`` move the tally, ``lane`` updates a worker row."""

    def __init__(self, parent: ScribeProgress, name: str) -> None:
        self._parent = parent
        self._name = name

    def start(self, total: int) -> None:
        self._parent._start(self._name, total)

    def advance(self, n: int = 1) -> None:
        self._parent._advance(self._name, n)

    def set_completed(self, done: int) -> None:
        self._parent._set_completed(self._name, done)

    def on_progress(self, done: int, total: int) -> None:
        """Adapt the ``_caption_all``/``_summarize_all`` ``on_progress`` contract
        — ``(0, total)`` to reveal, then ``(done, total)`` per item — onto this
        phase, so those passes need no bespoke closure in the caller."""
        if done == 0:
            self.start(total)
        else:
            self.set_completed(done)

    def lane(self, key: object, item: str, stage: str) -> None:
        self._parent._lane_update(key, item, stage)
