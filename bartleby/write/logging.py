from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.base import BaseCallbackHandler
from rich.console import Console

from bartleby.lib.consts import DEFAULT_MAX_RECURSIONS


def _parse_possible_json(value: Any) -> Any:
    """Attempt to parse string payloads containing JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _count_search_results(outputs: Any) -> int:
    """Best-effort extraction of a result count from tool outputs."""
    data = _parse_possible_json(outputs)

    if isinstance(data, list):
        return len(data)

    if isinstance(data, dict):
        for key in ("original_count", "total_items", "count"):
            count = data.get(key)
            if isinstance(count, int):
                return count

        for key in ("results", "items", "chunks", "data", "preview"):
            collection = data.get(key)
            if isinstance(collection, list):
                return len(collection)

    return 0


def _simple_summary(tool_name: str, inputs: Any, outputs: Any) -> tuple[str, str]:
    """
    Generate simple summary without LLM call.

    Args:
        tool_name: Name of the tool that was called
        inputs: Tool inputs
        outputs: Tool outputs

    Returns:
        Tuple of (short_summary, long_summary)
    """
    # Handle search tools
    if tool_name == "search_documents_fts":
        count = _count_search_results(outputs)
        query = inputs.get("query", "") if isinstance(inputs, dict) else ""
        short = f"Found {count} results"
        long = f"Full-text search for '{query}' returned {count} matching chunks"
        return short, long

    elif tool_name == "search_documents_semantic":
        count = _count_search_results(outputs)
        query = inputs.get("query", "") if isinstance(inputs, dict) else ""
        short = f"Found {count} results"
        long = f"Semantic search for '{query}' returned {count} similar chunks"
        return short, long

    elif tool_name == "get_chunk_window":
        chunk_id = inputs.get("chunk_id", "") if isinstance(inputs, dict) else ""
        short = "Read passage"
        long = f"Retrieved chunk window for {chunk_id} with surrounding context"
        return short, long

    elif tool_name == "get_full_document":
        doc_id = inputs.get("document_id", "") if isinstance(inputs, dict) else ""
        short = "Read document"
        long = f"Retrieved chunks from document {doc_id}"
        return short, long

    elif tool_name == "delegate_search":
        task = inputs.get("task", "") if isinstance(inputs, dict) else ""
        short = "Delegated search"
        long = f"Delegated search task: {task[:100]}"
        return short, long

    elif tool_name == "read_findings":
        short = "Read findings"
        long = "Retrieved all research findings for synthesis"
        return short, long

    # Default fallback
    short = f"Called {tool_name}"
    long = str(outputs)[:200] if outputs else f"Executed {tool_name}"
    return short, long


class StreamingLogger:
    """
    Manages streaming display of agent activity with phase tracking and token counts.
    Provides a clean, live-updating display for the CLI.
    """

    # Tool display messages for tool calls (shown briefly before result)
    TOOL_MESSAGES = {
        "search_documents_fts": "Searching text...",
        "search_documents_semantic": "Searching vectors...",
        "get_full_document": "Reading document...",
        "get_chunk_window": "Reading passage...",
        "manage_todo_tool": "Managing to-dos...",
        "delegate_search": "Delegating to Search Agent...",
        "read_findings": "Reading research findings...",
    }

    def __init__(
        self,
        console: Console,
        llm: BaseLanguageModel,
        log_path: Path,
        max_recursions: int = DEFAULT_MAX_RECURSIONS,
        todo_list=None,
        token_callback: BaseCallbackHandler | None = None,
        agent_name: str = "Agent",
        run_uuid: str | None = None,
    ):
        """
        Initialize streaming logger.

        Args:
            console: Rich console for output
            llm: Language model for generating summaries
            log_path: Path to log file
            max_recursions: Maximum recursions for the agent
            todo_list: TodoList instance for accessing todo list
            agent_name: Name of the agent being logged (for multi-agent systems)
            run_uuid: Unique ID for this agent run, stored with each log entry
        """
        self.console = console
        self.llm = llm
        self.log_path = log_path
        self.max_recursions = max_recursions
        self.todo_list = todo_list
        self.agent_name = agent_name
        self.run_uuid = run_uuid

        # State tracking
        self.recursion = 0
        self.current_tool_name = ""
        self.current_summary = ""
        self.action_history = []  # List of last 4 actions: [{tool_name, summary}, ...]

        # Search Agent progress tracking
        self.search_count = 0
        self.max_searches = 0

        # Token tracking (cumulative)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Pending tool calls
        self.pending_tools: Dict[str, Dict[str, Any]] = {}

        # Keep reference to token counter for formatted stats
        self.token_counter = token_callback
        self._last_ai_message_signature = None

    def on_recursion_start(self):
        """Called when agent starts a new graph recursion (super-step)."""
        self.recursion += 1

    def sync_recursion(self, step: int):
        """Ensure recursion counter reflects LangGraph's reported step."""
        if step > self.recursion:
            self.recursion = step

    def on_ai_message(self, message):
        """Handle a new AI message, deduplicating repeated streaming updates."""
        if self.agent_name != "Primary Agent":
            return

        signature = (
            getattr(message, "id", None)
            or getattr(message, "lc_id", None)
            or (getattr(message, "type", None), getattr(message, "content", None))
        )

        if signature == self._last_ai_message_signature:
            return

        self._last_ai_message_signature = signature
        self.on_recursion_start()

    def get_round(self) -> int:
        """Return the current primary agent round count."""
        return self.recursion

    def on_tool_call(self, tool_name: str, tool_id: str, args: Dict[str, Any]):
        """
        Store pending tool call.

        Args:
            tool_name: Name of the tool being called
            tool_id: Unique ID for this tool call
            args: Arguments passed to the tool
        """
        self.pending_tools[tool_id] = {
            'name': tool_name,
            'args': args,
            'recursion': self.recursion
        }

    def on_tool_result(self, tool_id: str, result: str, token_counter) -> Dict[str, Any]:
        """
        Process tool result:
        1. Generate summary (1 sentence + 1-2 paragraphs)
        2. Write to log file
        3. Update display state (current/previous)
        4. Return display data for Live widget

        Args:
            tool_id: Unique ID for this tool call
            result: Content returned by the tool
            token_counter: TokenCounterCallback instance

        Returns:
            Display data dict for rendering
        """
        if tool_id not in self.pending_tools:
            return self.get_display_data()

        # Get tool info
        tool_info = self.pending_tools.pop(tool_id)

        # Update token counts directly from counter
        self.total_input_tokens = token_counter.prompt_tokens
        self.total_output_tokens = token_counter.completion_tokens
        self.token_counter = token_counter  # Keep reference for formatted stats

        if tool_info['name'] == "manage_todo_tool":
            short, long = self._summarize_todo_tool(tool_info, result)
        else:
            short, long = _simple_summary(
                tool_info['name'],
                tool_info['args'],
                result,
            )

        # Write to log
        self._write_log(tool_info, result, short, long)

        # Update action history, keeping most recent items first
        if self.current_summary and self.current_tool_name:
            self.action_history.insert(0, {
                'tool_name': self.current_tool_name,
                'summary': self.current_summary
            })
            self.action_history = self.action_history[:4]

        self.current_tool_name = tool_info['name']
        self.current_summary = short

        return self.get_display_data()

    def get_friendly_name(self, tool_name: str) -> str:
        """
        Get friendly display name for a tool.

        Args:
            tool_name: Internal tool name

        Returns:
            Friendly name with emoji (e.g., "🔍 Full-text search")
        """
        return self.TOOL_MESSAGES.get(tool_name, tool_name)

    def get_display_data(self) -> Dict[str, Any]:
        """
        Return dict for Live widget to render.

        Returns:
            Dict with round, tokens, action history, current action, agent info
        """
        # Build action history with friendly names
        history_with_friendly_names = [
            {
                'friendly_name': self.get_friendly_name(action['tool_name']),
                'summary': ""
            }
            for action in self.action_history
        ]

        # Get formatted token stats
        token_stats_formatted = self.token_counter.get_stats() if self.token_counter else "[↑0/↓0/+0]"

        # Get todos from todo_list
        todos = []
        if self.todo_list:
            todos = self.todo_list.get_all_todos()

        search_count = self.search_count
        max_searches = self.max_searches

        if max_searches:
            search_count = min(search_count, max_searches)

        return {
            'agent_name': self.agent_name,
            'round': self.recursion,
            'max_rounds': self.max_recursions,
            'search_count': search_count,
            'max_searches': max_searches,
            'action_history': history_with_friendly_names,
            'current_action': {
                'friendly_name': self.get_friendly_name(self.current_tool_name) if self.current_tool_name else "",
                'summary': self.current_summary
            },
            'token_stats_formatted': token_stats_formatted,
            'todos': todos
        }

    def _write_log(self, tool_info: Dict[str, Any], result: Any, short: str, long: str):
        """
        Write entry to log.json with all context.

        Args:
            tool_info: Dict with tool name, args, iteration
            result: Tool result content
            short: Short summary (1 sentence)
            long: Long summary (1-2 paragraphs)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_uuid": self.run_uuid,
            "agent_name": self.agent_name,
            "round": tool_info['recursion'],
            "tool_name": tool_info['name'],
            "inputs": tool_info['args'],
            "outputs_summary": self._prepare_output_for_log(result),
            "summary_short": short,
            "summary_long": long,
            "tokens_at_entry": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens
            }
        }

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, separators=(",", ":")) + "\n")

    def _prepare_output_for_log(self, output: Any) -> Any:
        """
        Format tool output for log consumption.

        Attempts to keep structured data as JSON when possible.
        """
        if isinstance(output, (dict, list)):
            return self._truncate_structured_output(output)

        if isinstance(output, str):
            parsed = None
            try:
                parsed = json.loads(output)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                return self._truncate_structured_output(parsed)
            return output[:500]

        return str(output)[:500]

    def _truncate_structured_output(self, data: Any) -> Any:
        """
        Avoid logging extremely large payloads by returning previews.
        """
        if isinstance(data, list):
            if len(data) > 5:
                return {
                    "preview": data[:5],
                    "truncated": True,
                    "total_items": len(data)
                }
            return data

        if isinstance(data, dict):
            if len(data) <= 10:
                return data
            truncated = {}
            for idx, (key, value) in enumerate(data.items()):
                if idx >= 10:
                    truncated["__truncated__"] = True
                    truncated["__total_keys__"] = len(data)
                    break
                truncated[key] = value
            return truncated

        return data

    def _summarize_todo_tool(self, tool_info: Dict[str, Any], result: Any) -> tuple[str, str]:
        """Return a lightweight summary for todo list actions without LLM calls."""
        action = tool_info.get('args', {}).get('action', 'list')
        short = f"manage_todo_tool ({action})"

        if isinstance(result, dict):
            message = result.get("message")
            todo = result.get("todo")
            if message and todo:
                long = f"{message}: {todo.get('task', '')} [{todo.get('status', '')}]"
            elif message:
                long = message
            else:
                long = json.dumps(result)[:200]
        elif isinstance(result, str):
            long = result[:200]
        else:
            long = str(result)[:200]

        return short, long
