"""Main entry point for the write (research agent) command."""

import os
import time
import uuid
from pathlib import Path

from rich.prompt import Prompt
from rich.console import Console
from rich.live import Live
from rich.text import Text
from langchain_core.messages import AIMessage, ToolMessage

# Disable tokenizers parallelism warning (set before importing sentence_transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover - fallback when langgraph not installed
    class GraphRecursionError(RuntimeError):
        """Placeholder when LangGraph isn't available."""
        pass

from bartleby.lib.console import send
from bartleby.lib.consts import (
    DEFAULT_MAX_RECURSIONS,
    DEFAULT_MAX_SEARCH_OPERATIONS,
    DEFAULT_MAX_TODO_ROUNDS,
    DEFAULT_MAX_TOTAL_ROUNDS,
    DEFAULT_MAX_TOTAL_TOKENS,
    EMBEDDING_MODEL,
)
from bartleby.lib.utils import load_config, load_llm_from_config
from bartleby.write.primary_agent import run_primary_agent
from bartleby.write.logging import StreamingLogger
from bartleby.write.token_counter import TokenCounterCallback


def format_display(display_data: dict) -> Text:
    """
    Render display data as plain text (no panel border).

    Args:
        display_data: Dict with agent_name, round, search_count, action_history, current_action, token_stats_formatted

    Returns:
        Rich Text object for Live display
    """
    lines: list[Text] = []

    round_text = f"Primary Agent: round {display_data['round']}/{display_data['max_rounds']}"
    lines.append(Text(round_text, style="bold"))

    # Nested Search Agent status when active
    max_searches = display_data.get('max_searches', 0)
    if max_searches:
        search_count = display_data.get('search_count', 0)
        search_text = f" ∟ Search Agent: round {search_count}/{max_searches}"
        lines.append(Text(search_text, style="cyan"))

    # Todo list
    todos = display_data.get('todos', [])
    if todos:
        lines.append(Text(""))
        lines.append(Text("To-do list:", style="bold"))
        for todo in todos:
            symbol = "☐"
            style = "white"
            if todo['status'] == 'complete':
                symbol = "☑"
                style = "dim"
            elif todo['status'] == 'active':
                symbol = "☐"
                style = "bold cyan"

            lines.append(Text(f"{symbol} {todo['task']}", style=style))

    # Actions list
    has_actions = display_data['action_history'] or display_data['current_action']['summary']
    if has_actions:
        lines.append(Text(""))
        lines.append(Text("Actions list:", style="bold"))

        # Action history (dim)
        for action in display_data['action_history']:
            lines.append(Text(
                f"- {action['friendly_name']}",
                style="dim"
            ))

        # Current action (bold)
        current = display_data['current_action']
        if current['summary']:
            lines.append(Text(
                f"- {current['friendly_name']}",
                style="bold"
            ))

    # Token stats
    lines.append(Text(""))
    token_text = display_data['token_stats_formatted']
    lines.append(Text(f"Tokens: {token_text}", style="white"))

    return Text("\n").join(lines)


def main(db_path: Path):
    """
    Run the research agent.

    Args:
        db_path: Path to the document database
    """
    # Load config from ~/.bartleby/config.yaml
    config = load_config()

    def _coerce_positive_int(value, default):
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

    def _coerce_non_negative_int(value, default):
        try:
            parsed = int(value)
            return parsed if parsed >= 0 else default
        except (TypeError, ValueError):
            return default

    def _coerce_non_negative_optional(value):
        try:
            if value is None:
                return None
            parsed = int(value)
            return parsed if parsed >= 0 else None
        except (TypeError, ValueError):
            return None

    # Generate unique run ID
    run_uuid = uuid.uuid4().hex[:8]

    # Set up the bartleby folder
    bartleby_dir_path = db_path.parent / ".bartleby"
    bartleby_dir_path.mkdir(parents=True, exist_ok=True)

    # Create findings directory
    findings_dir = bartleby_dir_path / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    # Clean up previous run files
    log_path = bartleby_dir_path / "log.json"
    todos_path = bartleby_dir_path / "todos.json"
    for p in [log_path, todos_path]:
        if p.exists():
            p.unlink()

    send(message_type="SPLASH")
    send("Starting Bartleby research agent", "BIG")

    # Validate database exists
    if not db_path.exists():
        send(f"Database not found: {db_path}", "ERROR")
        send("Please run 'bartleby read' first to process documents.", "WARN")
        return

    # Load LLM from config
    send("Loading LLM from configuration...", "BIG")
    llm = load_llm_from_config(config)
    if llm is None:
        send("No LLM configured", "ERROR")
        send("Please run 'bartleby ready' first to configure your LLM.", "WARN")
        return

    # Load embedding model
    send(f"Loading embedding model: {EMBEDDING_MODEL}", "BIG")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    send("", "EMPTY")

    # Ask user for direction
    send("What would you like me to investigate?", "BIG")
    user_direction = Prompt.ask("Investigation direction")

    if not user_direction or not user_direction.strip():
        send("No direction provided. Exiting.", "WARN")
        return

    send("\n", "EMPTY")

    # Initialize shared resources
    model_name = config.get("model", "")

    # Load limits with safe defaults
    max_todo_rounds = _coerce_positive_int(
        config.get("max_todo_rounds"), DEFAULT_MAX_TODO_ROUNDS
    )
    max_total_rounds = _coerce_positive_int(
        config.get("max_total_rounds"), DEFAULT_MAX_TOTAL_ROUNDS
    )
    if max_total_rounds < max_todo_rounds + 1:
        max_total_rounds = max_todo_rounds + 1
        send(
            f"Adjusted max_total_rounds to {max_total_rounds} so there is at least one round to write the report.",
            "WARN"
        )
    max_search_operations = _coerce_positive_int(
        config.get("max_search_operations"), DEFAULT_MAX_SEARCH_OPERATIONS
    )
    token_budget_limit = _coerce_non_negative_optional(
        config.get("token_budget", DEFAULT_MAX_TOTAL_TOKENS)
    )
    if token_budget_limit is None:
        send(
            "No token budget limit set in config. Agent will run without token budget constraints. "
            "To set a budget, add 'token_budget: <number>' to ~/.bartleby/config.yaml",
            "WARN"
        )

    console = Console(force_terminal=True, force_interactive=True)

    # Token tracking
    token_counter = TokenCounterCallback(model_name=model_name)
    token_counter.max_recursions = max_total_rounds
    token_counter.token_budget = token_budget_limit

    # Logging setup - load TodoList for logger to access
    from bartleby.write.memory import TodoList
    todo_list = TodoList(str(todos_path))

    logger = StreamingLogger(
        console,
        llm,
        log_path,
        max_recursions=max_total_rounds,
        todo_list=todo_list,
        token_callback=token_counter,
        agent_name="Primary Agent",
        run_uuid=run_uuid,
    )

    final_report_text = ""

    def _run_agent():
        """Run the Primary Agent and return final report."""
        nonlocal final_report_text

        initial_display = format_display(logger.get_display_data())
        last_rendered = {"signature": initial_display.plain}
        last_render_time = {"time": 0}

        def render_display(data: dict):
            rendered = format_display(data)
            signature = rendered.plain
            current_time = time.time()

            # Only render if content changed AND enough time has passed
            if signature != last_rendered["signature"]:
                # Debounce: require 100ms between renders
                if current_time - last_render_time["time"] >= 0.1:
                    last_rendered["signature"] = signature
                    last_render_time["time"] = current_time
                    live.update(rendered)
                    live.refresh()

        with Live(initial_display, console=console, refresh_per_second=4, auto_refresh=False) as live:
            # Define display update callback
            def update_display():
                render_display(logger.get_display_data())

            for result in run_primary_agent(
                user_direction=user_direction,
                llm=llm,
                db_path=db_path,
                findings_dir=findings_dir,
                todos_path=todos_path,
                embedding_model=embedding_model,
                run_uuid=run_uuid,
                token_counter=token_counter,
                max_todo_rounds=max_todo_rounds,
                max_total_rounds=max_total_rounds,
                max_search_operations=max_search_operations,
                logger=logger,
                display_callback=update_display,
            ):
                if "chunk" in result:
                    chunk = result["chunk"]

                    if "messages" in chunk and len(chunk["messages"]) > 0:
                        last_message = chunk["messages"][-1]

                        # Handle AI messages (increment round counter)
                        if isinstance(last_message, AIMessage):
                            logger.on_ai_message(last_message)
                            render_display(logger.get_display_data())

                        # Handle tool calls from AI messages
                        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                tool_name = tool_call.get('name', '')
                                tool_call_id = tool_call.get('id', '')
                                tool_args = tool_call.get('args', {})
                                logger.on_tool_call(tool_name, tool_call_id, tool_args)

                        # Handle tool results
                        if isinstance(last_message, ToolMessage):
                            tool_call_id = getattr(last_message, 'tool_call_id', None)
                            if tool_call_id:
                                display_data = logger.on_tool_result(
                                    tool_call_id,
                                    last_message.content,
                                    token_counter
                                )
                                render_display(display_data)

                if "agent" in result:
                    final_report_text = result["agent"].get("output", "")
                    live.stop()
                    break

    try:
        _run_agent()

        aggregate_stats = token_counter.get_stats()

    except KeyboardInterrupt:
        send("\n\nResearch interrupted by user.", "WARN")
        send(f"{token_counter.get_stats()}", "TOKENS")
        return
    except GraphRecursionError:
        send("Agent reached the recursion limit before completing the report.", "ERROR")
        send(f"{token_counter.get_stats()}", "TOKENS")
        return
    except Exception as e:
        send(f"Error during research: {e}", "ERROR")
        send(f"{token_counter.get_stats()}", "TOKENS")
        raise

    if not final_report_text:
        send("Agent did not produce a final report.", "ERROR")
        send(f"{aggregate_stats}", "TOKENS")
        return

    send(message_type="REPORT_BORDER_TOP")
    send("✎ RESEARCH REPORT", "REPORT_TITLE")
    send(message_type="REPORT_BORDER_BOTTOM")
    send(final_report_text, "REPORT")

    report_dir = db_path.parent / ".bartleby"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    with report_path.open("w", encoding="utf-8") as f:
        f.write(final_report_text)

    send(f"Report saved to: {report_path}", "COMPLETE")
    send(f"Final: {logger.recursion} rounds | {aggregate_stats}", "INFO_DIM")
