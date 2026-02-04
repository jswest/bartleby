"""Live progress display for the write command."""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from rich.console import Console, ConsoleOptions, RenderResult
from rich.spinner import Spinner
from rich.text import Text

from bartleby.write.logging import TOOL_MESSAGES


COMPLETED_LABELS = {
    "search_documents_fts": "Searched text",
    "search_documents_semantic": "Searched vectors",
    "get_full_document": "Read document",
    "get_chunk_window": "Read passage",
    "list_documents": "Listed documents",
    "get_document_summary": "Read summary",
    "summarize_document": "Summarized document",
    "read_notes": "Read notes",
    "save_note": "Saved note",
    "write_file": "Wrote file",
}


def _list_count(data) -> Optional[int]:
    """Extract item count from a list or a truncated-result dict."""
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and data.get("truncated"):
        count = data.get("original_count")
        if count is not None:
            return count
    return None


def _extract_summary(tool_name: str, observations: str) -> str:
    """Extract a brief result indicator from tool output."""
    try:
        data = json.loads(observations)
    except (json.JSONDecodeError, TypeError):
        return ""

    if tool_name in ("search_documents_fts", "search_documents_semantic"):
        count = _list_count(data)
        if count is not None:
            return f"({count} results)"
        if isinstance(data, dict) and "error" in data:
            return "(no results)"
        return ""

    if tool_name == "get_chunk_window":
        if isinstance(data, dict):
            n = data.get("returned_chunks")
            if n is not None:
                return f"({n} chunks)"
        return ""

    if tool_name == "get_full_document":
        if isinstance(data, dict):
            parts = []
            fname = data.get("origin_file_path")
            if fname:
                parts.append(os.path.basename(fname))
            rc = data.get("returned_chunks")
            if rc is not None:
                parts.append(f"{rc} chunks")
            return f"({', '.join(parts)})" if parts else ""
        return ""

    if tool_name == "list_documents":
        count = _list_count(data)
        if count is not None:
            return f"({count} documents)"
        return ""

    if tool_name in ("get_document_summary", "summarize_document"):
        if isinstance(data, dict):
            title = data.get("title", "")
            if title:
                short = title[:40] + "..." if len(title) > 40 else title
                return f"({short})"
        return ""

    if tool_name == "save_note":
        if isinstance(data, dict):
            msg = data.get("message", "")
            return f"({msg})" if msg else ""
        return ""

    if tool_name == "write_file":
        if isinstance(data, dict):
            fp = data.get("filepath", "")
            if fp:
                return f"({os.path.basename(fp)})"
        return ""

    return ""


class ProgressDisplay:
    """Rich renderable showing completed tool-call log lines above an active spinner.

    Usage::

        progress = ProgressDisplay()
        with Live(progress, console=console, refresh_per_second=4):
            progress.start_tool(tool_name)
            ...
            progress.complete_tool(tool_name, observations)
    """

    def __init__(self) -> None:
        self.spinner = Spinner("dots", text="Thinking...")
        self._completed_lines: List[Text] = []
        self._step_start: float = time.monotonic()

    def start_tool(self, tool_name: str) -> None:
        """Update the spinner text for an active tool call."""
        label = TOOL_MESSAGES.get(tool_name, tool_name)
        self.spinner.text = label

    def complete_tool(self, tool_name: str, observations: str) -> None:
        """Finalize a tool call and append a log line."""
        now = time.monotonic()
        elapsed = now - self._step_start
        self._step_start = now

        label = COMPLETED_LABELS.get(tool_name, tool_name)
        summary = _extract_summary(tool_name, observations)

        left = f"  [green]\u2713[/green] {label}"
        if summary:
            left += f" {summary}"

        time_str = f"{elapsed:.1f}s"

        # Approximate visible length for dot-fill alignment.
        visible_left = len(label) + (len(f" {summary}") if summary else 0) + 4
        target_width = 58
        dots_needed = max(2, target_width - visible_left - len(time_str))
        dots = " " + "." * dots_needed + " "

        line = Text.from_markup(f"{left}[dim]{dots}{time_str}[/dim]")
        self._completed_lines.append(line)

        self.spinner.text = "Thinking..."

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Yield completed log lines followed by the spinner."""
        for line in self._completed_lines:
            yield line
        yield self.spinner
