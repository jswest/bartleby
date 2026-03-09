"""Logs agent tool calls to a JSON-lines file."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict


TOOL_MESSAGES = {
    "search_expert": "Searching corpus...",
    "search_documents": "Searching documents...",
    "get_full_document": "Reading document...",
    "get_chunk_window": "Reading passage...",
    "list_documents": "Listing documents...",
    "get_document_summary": "Reading summary...",
    "summarize_document": "Summarizing document...",
    "read_notes": "Reading research findings...",
    "save_note": "Saving note...",
    "write_file": "Writing file...",
    "write_csv": "Writing CSV...",
    "describe_self": "Describing capabilities...",
    "request_more_steps": "Requesting more steps...",
}


def _truncate_for_log(output: Any, max_chars: int = 500) -> Any:
    """Truncate tool output for log consumption."""
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except Exception:
            return output[:max_chars]
        output = parsed

    if isinstance(output, list) and len(output) > 5:
        return {"preview": output[:5], "truncated": True, "total_items": len(output)}

    if isinstance(output, dict) and len(output) > 10:
        truncated = dict(list(output.items())[:10])
        truncated["__truncated__"] = True
        truncated["__total_keys__"] = len(output)
        return truncated

    return output


class StreamingLogger:
    """Logs agent tool calls to a per-session JSON-lines file."""

    def __init__(self, session_dir: Path, token_counter=None):
        self.log_path = session_dir / "log.jsonl"
        self.token_counter = token_counter
        self.pending_tools: Dict[str, Dict[str, Any]] = {}

    def on_tool_call(self, tool_name: str, tool_id: str, args: Dict[str, Any]):
        """Store pending tool call."""
        self.pending_tools[tool_id] = {"name": tool_name, "args": args}

    def on_tool_result(self, tool_id: str, result: str, token_counter=None):
        """Process tool result and write log entry."""
        if tool_id not in self.pending_tools:
            return

        tool_info = self.pending_tools.pop(tool_id)

        if token_counter:
            self.token_counter = token_counter

        self._write_log(tool_info, result)

    def get_friendly_name(self, tool_name: str) -> str:
        return TOOL_MESSAGES.get(tool_name, tool_name)

    def _write_log(self, tool_info: Dict[str, Any], result: Any):
        """Write entry to log.jsonl."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_info["name"],
            "inputs": tool_info["args"],
            "outputs_summary": _truncate_for_log(result),
        }

        if self.token_counter:
            log_entry["tokens"] = {
                "input": self.token_counter.prompt_tokens,
                "output": self.token_counter.completion_tokens,
            }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
