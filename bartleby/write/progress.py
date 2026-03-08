"""Live progress display for the write command."""

from __future__ import annotations

import json
import os
import time

from rich.console import Console, ConsoleOptions, RenderResult
from rich.spinner import Spinner
from rich.text import Text

from bartleby.write.logging import TOOL_MESSAGES


COMPLETED_LABELS = {
    "search_expert": "Searched corpus",
    "search_documents": "Searched documents",
    "get_full_document": "Read document",
    "get_chunk_window": "Read passage",
    "list_documents": "Listed documents",
    "get_document_summary": "Read summary",
    "summarize_document": "Summarized document",
    "read_notes": "Read notes",
    "save_note": "Saved note",
    "write_file": "Wrote file",
}


def _describe_tool_start(tool_name: str, tool_args: dict) -> str:
    """Build a descriptive message from tool name and arguments."""
    if tool_name == "search_expert":
        task = tool_args.get("task", "")
        if task:
            short = task[:60] + "..." if len(task) > 60 else task
            return f'Researching: "{short}"'
        return "Searching corpus..."

    if tool_name == "search_documents":
        query = tool_args.get("query", "")
        if query:
            short = query[:50] + "..." if len(query) > 50 else query
            return f'Searching for "{short}"'
        return "Searching documents..."

    if tool_name == "get_chunk_window":
        return "Reading passage in context..."

    if tool_name == "get_full_document":
        doc_id = tool_args.get("document_id", "")
        if doc_id:
            short = doc_id[:20] + "..." if len(doc_id) > 20 else doc_id
            return f"Reading document {short}..."
        return "Reading document..."

    if tool_name in ("get_document_summary", "summarize_document"):
        return "Reading document summary..."

    return TOOL_MESSAGES.get(tool_name, f"{tool_name}...")


def extract_tool_summary(tool_name: str, observations: str) -> str:
    """Extract a brief result indicator from tool output."""
    try:
        data = json.loads(observations)
    except (json.JSONDecodeError, TypeError):
        return ""

    if tool_name == "search_documents":
        count = _list_count(data)
        if count is not None:
            return f"{count} results"
        if isinstance(data, dict) and "error" in data:
            return "no results"
        return ""

    if tool_name == "search_expert":
        # Subagent returns a string, not JSON
        return ""

    if tool_name == "get_chunk_window":
        if isinstance(data, dict):
            n = data.get("returned_chunks")
            if n is not None:
                return f"{n} chunks"
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
            return ", ".join(parts) if parts else ""
        return ""

    if tool_name == "list_documents":
        count = _list_count(data)
        if count is not None:
            return f"{count} documents"
        return ""

    if tool_name in ("get_document_summary", "summarize_document"):
        if isinstance(data, dict):
            title = data.get("title", "")
            if title:
                short = title[:40] + "..." if len(title) > 40 else title
                return short
        return ""

    if tool_name == "save_note":
        if isinstance(data, dict):
            return data.get("message", "")
        return ""

    if tool_name == "write_file":
        if isinstance(data, dict):
            fp = data.get("filepath", "")
            if fp:
                return os.path.basename(fp)
        return ""

    return ""


def _list_count(data) -> int | None:
    """Extract item count from a list or a truncated-result dict."""
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and data.get("truncated"):
        count = data.get("original_count")
        if count is not None:
            return count
    return None


class ProgressDisplay:
    """Rich renderable showing step-by-step agent progress.

    Displays:
        Step 2/10 | Searching for "PM2.5 equity"
               +  | Found 3 passages across 2 documents (0.2s)
        Step 3/10 | Thinking...
    """

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps
        self.spinner = Spinner("dots", text="Thinking...")
        self._completed_lines: list[Text] = []
        self._step_start: float = time.monotonic()
        self._step_num: int = 0

    def start_tool(self, tool_name: str, tool_args: dict | None = None) -> None:
        """Update the spinner text for an active tool call."""
        self._step_num += 1
        self._step_start = time.monotonic()
        description = _describe_tool_start(tool_name, tool_args or {})
        step_label = f"Step {self._step_num}/{self.max_steps}"
        self.spinner.text = Text.from_markup(
            f"[bold]{step_label}[/bold] [dim]|[/dim] {description}"
        )

    def complete_tool(self, tool_name: str, observations: str) -> None:
        """Finalize a tool call and append a log line."""
        now = time.monotonic()
        elapsed = now - self._step_start
        self._step_start = now

        label = COMPLETED_LABELS.get(tool_name, tool_name)
        summary = extract_tool_summary(tool_name, observations)

        step_label = f"Step {self._step_num}/{self.max_steps}"
        time_str = f"({elapsed:.1f}s)"

        # Completed step line
        left = f"[bold]{step_label}[/bold] [dim]|[/dim] [green]+[/green] {label}"
        if summary:
            left += f" [dim]- {summary}[/dim]"
        left += f" [dim]{time_str}[/dim]"

        line = Text.from_markup(left)
        self._completed_lines.append(line)

        self.spinner.text = "Thinking..."

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Yield completed log lines followed by the spinner."""
        for line in self._completed_lines:
            yield line
        yield self.spinner
