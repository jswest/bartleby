from __future__ import annotations

"""Tool logging and display with LLM summarization."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING
import math

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from rich.console import Console

from bartleby.lib.consts import DEFAULT_MAX_RECURSIONS


def summarize_with_llm(
    llm: BaseLanguageModel,
    tool_name: str,
    inputs: Any,
    outputs: Any,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, str]:
    """
    Generate one-sentence and one-paragraph summaries using LLM.

    Args:
        llm: Language model to use for summarization
        tool_name: Name of the tool that was called
        inputs: Tool inputs (will be converted to string)
        outputs: Tool outputs (will be converted to string)

    Returns:
        Tuple of (short_summary, long_summary)
        - short_summary: ONE sentence (max 100 chars)
        - long_summary: ONE to TWO paragraphs (max 400 chars)
    """
    # Convert inputs/outputs to strings, truncating if too long
    inputs_str = str(inputs)[:500]
    outputs_str = str(outputs)[:5000]

    prompt = f"""
# Summarize the rsults of this tool invocation concisely.

### Tool:

```
{tool_name}
```

### Inputs:

```
{inputs_str}
```

### Outputs:

```
{outputs_str}
```

## Provide

1. A one-sentence summary (max 75 characters) - factual, specific
2. A one-paragraph summary (max 400 characters) - detailed explanation of what was found

Format your response as:
SHORT: <one sentence>
LONG: <one to two paragraphs>
"""

    try:
        config = {"callbacks": callbacks} if callbacks else None
        response = llm.invoke([HumanMessage(content=prompt)], config=config)
        content = response.content.strip()

        # Parse response
        lines = content.split('\n')
        short = ""
        long = ""

        for line in lines:
            if line.startswith("SHORT:"):
                short = line.replace("SHORT:", "").strip()
            elif line.startswith("LONG:"):
                long = line.replace("LONG:", "").strip()

        return short or f"Called {tool_name}", long or short

    except Exception as e:
        # Fallback if LLM fails
        short = f"Called {tool_name}"
        long = f"Called {tool_name}, but summarization failed with {e}."
        return short, long


class StreamingLogger:
    """
    Manages streaming display of agent activity with phase tracking and token counts.
    Provides a clean, live-updating display for the CLI.
    """

    # Tool display messages for tool calls (shown briefly before result)
    TOOL_MESSAGES = {
        "search_documents_fts": "Searching text",
        "search_documents_semantic": "Searching vectors",
        "get_full_document": "Reading document",
        "get_chunk_window": "Reading passage",
        "append_to_scratchpad_tool": "Taking notes",
        "manage_todo_tool": "Managing to-dos",
        "delegate_search": "Delegating to Search Agent",
        "read_scratchpad_tool": "Reading notes",
    }

    def __init__(
            self,
            console: Console,
            llm: BaseLanguageModel,
            log_path: Path,
            max_recursions: int = DEFAULT_MAX_RECURSIONS,
            search_tools = None,
            token_callback: BaseCallbackHandler | None = None,
            agent_name: str = "Agent",
        ):
        """
        Initialize streaming logger.

        Args:
            console: Rich console for output
            llm: Language model for generating summaries
            log_path: Path to log file
            max_recursions: Maximum recursions for the agent
            search_tools: DocumentSearchTools instance for accessing todo list
            agent_name: Name of the agent being logged (for multi-agent systems)
        """
        self.console = console
        self.llm = llm
        self.log_path = log_path
        self.max_recursions = max_recursions
        self.search_tools = search_tools
        self.agent_name = agent_name

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
        self._suppress_recursions = False

    def on_recursion_start(self):
        """Called when agent starts a new graph recursion (super-step)."""
        self.recursion += 1

    def sync_recursion(self, step: int):
        """Ensure recursion counter reflects LangGraph's reported step."""
        if step > self.recursion:
            self.recursion = step

    def on_ai_message(self, message):
        """Handle a new AI message, deduplicating repeated streaming updates."""
        if self._suppress_recursions or self.agent_name != "Primary Agent":
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
            callbacks = [self.token_counter] if self.token_counter else None
            self._suppress_recursions = True
            try:
                short, long = summarize_with_llm(
                    self.llm,
                    tool_info['name'],
                    tool_info['args'],
                    result,
                    callbacks=callbacks,
                )
            finally:
                self._suppress_recursions = False

        # Write to log
        self._write_log(tool_info, result, short, long)

        # Update action history
        if self.current_summary and self.current_tool_name:
            # Add current to history
            self.action_history.append({
                'tool_name': self.current_tool_name,
                'summary': self.current_summary
            })
            # Keep only last 4
            if len(self.action_history) > 4:
                self.action_history = self.action_history[-4:]

        # Set new current action
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
                'summary': action['summary']
            }
            for action in self.action_history
        ]

        # Get formatted token stats
        token_stats_formatted = self.token_counter.get_stats() if self.token_counter else "[↑0/↓0/+0]"

        # Get todos from search_tools
        todos = []
        if self.search_tools:
            todos = self.search_tools.todo_list.get_all_todos()

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
            f.write(json.dumps(log_entry, indent=2) + "\n")

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
