"""Live progress view for the summarize matrix.

On a terminal: an overall matrix bar + a streaming active-model line + a pulse
bar + a per-model dashboard table. Off a TTY (piped/CI) it degrades to plain
stderr lines so logs stay readable.
"""

from __future__ import annotations

import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from bartleby.benchmark.sources import count_tokens


class BenchmarkProgress:
    """The object is its own renderable (``__rich__``); Live re-reads its
    mutable state on each refresh tick, so per-chunk callbacks only stash the
    latest text and the token estimate is computed at render time, not on
    every one of the hundreds of stream chunks."""

    def __init__(self, models: list[str], calls_per_model: int, total_calls: int):
        # isatty() is the honest signal — rich's is_terminal over-detects when
        # FORCE_COLOR/CI-style env vars are set, which would spray escape codes
        # into a redirected log. Animate only on a real terminal.
        self.tty = sys.stderr.isatty()
        self.console = Console(file=sys.stderr, force_terminal=self.tty)
        self.calls_per_model = calls_per_model  # runs × docs
        self.total = total_calls
        self.state = {m: {"n": 0, "tps": None, "passed": None, "running": False}
                      for m in models}
        self.active: dict | None = None

        self._overall = Progress(
            TextColumn("[bold]Summarizing"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(), TextColumn("calls"),
            TimeElapsedColumn(), TextColumn("• ETA"), TimeRemainingColumn(),
            console=self.console,
        )
        self._task = self._overall.add_task("calls", total=total_calls)
        self._pulse = Progress(
            TextColumn("   "),
            BarColumn(bar_width=40, pulse_style="cyan"),
            TextColumn("[dim]generating…"),
            console=self.console,
        )
        self._pulse.add_task("gen", total=None)  # total=None -> animated pulse
        self._live = Live(self, console=self.console, refresh_per_second=12)

    # -- rich integration --
    def __rich__(self):
        parts = [self._overall]
        if self.active is not None:
            parts += [self._active_line(), self._pulse]
        parts.append(self._table())
        return Group(*parts)

    def _active_line(self) -> Text:
        # Live estimate: tiktoken on the streamed text so far / wall time. This
        # is intentionally distinct from the table's Tok/s (Ollama's exact
        # eval_count / eval_duration, known only at completion) — this one shows
        # "tokens accruing now", that one shows measured throughput.
        a = self.active
        elapsed = time.perf_counter() - a["start"]
        toks = count_tokens(a["content"])
        tps = toks / elapsed if elapsed > 0 else 0
        return Text.assemble(
            ("▶ ", "cyan"),
            (f"{a['model']} · {a['doc']} (run {a['run']})", "bold"),
            (f"   ~{toks} tok · {tps:.0f} tok/s · {elapsed:.1f}s", "dim"),
        )

    def _table(self) -> Table:
        t = Table(box=None, pad_edge=False, expand=False)
        t.add_column("Model")
        t.add_column("Status")
        t.add_column("Runs", justify="right")
        t.add_column("Tok/s", justify="right")
        t.add_column("OK")  # all calls so far succeeded (parsed DocumentSummary)?
        for model, st in self.state.items():
            if st["running"]:
                status = Text("▶ running", style="cyan")
            elif st["n"] >= self.calls_per_model:
                status = Text("✓ done", style="green")
            elif st["n"]:
                status = Text("waiting", style="yellow")
            else:
                status = Text("queued", style="dim")
            tps = f"{st['tps']:.1f}" if st["tps"] else "—"
            if st["passed"] is None:
                ok_cell = Text("—", style="dim")
            elif st["passed"]:
                ok_cell = Text("ok", style="green")
            else:
                ok_cell = Text("FAIL", style="red")
            t.add_row(model, status, f"{st['n']}/{self.calls_per_model}", tps, ok_cell)
        return t

    # -- lifecycle --
    def __enter__(self):
        if self.tty:
            self._live.start()
        return self

    def __exit__(self, *exc):
        if self.tty:
            self._live.refresh()
            self._live.stop()

    def start_call(self, model: str, doc: str, run_idx: int, call_no: int) -> None:
        self.active = {"model": model, "doc": doc, "run": run_idx, "content": "",
                       "start": time.perf_counter()}
        self.state[model]["running"] = True
        if not self.tty:
            print(f"  [{call_no}/{self.total}] {model} · {doc} (run {run_idx})",
                  file=sys.stderr, flush=True)

    def on_chunk(self, content: str) -> None:
        if self.active is not None:
            self.active["content"] = content  # render reads this on its own tick

    def finish_call(self, model: str, ok: bool, tps: float | None) -> None:
        st = self.state[model]
        st["running"] = False
        st["n"] += 1
        st["passed"] = ok if st["passed"] is None else (st["passed"] and ok)
        if ok and tps:
            st["tps"] = tps
        self.active = None
        if self.tty:
            self._overall.advance(self._task)
        else:
            extra = f" · {tps:.0f} tok/s" if (ok and tps) else ""
            print(f"      {'✓' if ok else '✗'} {model}{extra}",
                  file=sys.stderr, flush=True)
