"""Main entry point for the write (research agent) command."""

import os
from pathlib import Path
from rich.prompt import Prompt
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown

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
    DEFAULT_AGENT_CONTEXT_TOKENS,
    DEFAULT_AGENT_RETRY_ATTEMPTS,
    DEFAULT_MAX_TOTAL_TOKENS,
    EMBEDDING_MODEL,
)
from bartleby.lib.utils import load_config, load_llm_from_config
from bartleby.write.agent import run_agent
from bartleby.write.logging import StreamingLogger
from bartleby.write.token_counter import TokenCounterCallback
from bartleby.write.tools import DocumentSearchTools


def format_display(display_data: dict) -> Text:
    """
    Render display data as plain text (no panel border).

    Args:
        display_data: Dict with phase, iteration, action_history, current_action, token_stats_formatted

    Returns:
        Rich Text object for Live display
    """
    lines = []

    # Line 1: Phase and recursion
    lines.append(Text(
        f"({display_data['recursion']}/{display_data['max_recursions']} iterations used)"
    ))

    # Blank line
    lines.append(Text(""))

    # Todo list section (if todos exist)
    todos = display_data.get('todos', [])
    if todos:
        lines.append(Text("(to-do list)", style="dim"))
        for todo in todos:
            # Use checkbox symbols based on status
            if todo['status'] == 'complete':
                symbol = "☑"
                style = "bold dim"
            elif todo['status'] == 'active':
                symbol = "☐"
                style = "bold cyan"
            else:  # pending
                symbol = "☐"
                style = "dim"

            lines.append(Text(f"{symbol} {todo['task']}", style=style))

        # Blank line after todos
        lines.append(Text(""))

    # Historical actions (up to 4, shown dim)
    for action in display_data['action_history']:
        lines.append(Text(
            f"{action['friendly_name']} {action['summary']}",
            style="dim"
        ))

    # Current action (shown bold)
    current = display_data['current_action']
    if current['summary']:
        lines.append(Text(
            f"{current['friendly_name']} {current['summary']}",
            style="bold cyan"
        ))
    else:
        lines.append(Text("Now: Initializing...", style="dim"))

    # Blank line
    lines.append(Text(""))

    # Token stats
    lines.append(Text(display_data['token_stats_formatted'], style="white"))

    # Combine all lines
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

    # Set up the bartleby folder.
    bartleby_dir_path = db_path.parent / ".bartleby"
    bartleby_dir_path.mkdir(parents=True, exist_ok=True)

    # Clean it up.
    log_path = bartleby_dir_path / "log.json"
    scratchpad_path = bartleby_dir_path / "scratchpad.md"
    todos_path = bartleby_dir_path / "todos.json"
    for p in [log_path, scratchpad_path, todos_path]:
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

    # Initialize token counter with model name from config
    model_name = config.get("model", "")
    token_counter = TokenCounterCallback(model_name=model_name)

    # Initialize tools
    search_tools = DocumentSearchTools(db_path, scratchpad_path, todos_path, embedding_model)

    # Load limits with safe defaults
    max_recursions = _coerce_positive_int(
        config.get("max_recursions"), DEFAULT_MAX_RECURSIONS
    )
    context_token_limit = _coerce_positive_int(
        config.get("max_context_tokens"), DEFAULT_AGENT_CONTEXT_TOKENS
    )
    agent_retry_attempts = _coerce_non_negative_int(
        config.get("agent_retry_attempts"), DEFAULT_AGENT_RETRY_ATTEMPTS
    )
    token_budget_limit = _coerce_non_negative_optional(
        config.get("token_budget", DEFAULT_MAX_TOTAL_TOKENS)
    )

    # Create console and logger (pass search_tools for todo list access)
    console = Console()
    logger = StreamingLogger(
        console,
        llm,
        log_path,
        max_recursions=max_recursions,
        search_tools=search_tools
    )

    def _budget_status():
        data = {
            "recursions_used": logger.recursion,
            "recursion_limit": max_recursions,
            "tokens_used": token_counter.total_tokens,
        }
        if token_budget_limit is not None:
            data["token_budget"] = token_budget_limit
            data["tokens_remaining_estimate"] = max(token_budget_limit - token_counter.total_tokens, 0)
        return data

    try:
        with Live(format_display(logger.get_display_data()), console=console, refresh_per_second=4) as live:
            # Run agent with streaming
            for result in run_agent(
                user_direction=user_direction,
                llm=llm,
                token_counter=token_counter,
                search_tools=search_tools,
                max_recursions=max_recursions,
                context_token_limit=context_token_limit,
                model_retry_attempts=agent_retry_attempts,
                token_budget_limit=token_budget_limit,
                budget_status_getter=_budget_status,
            ):
                # Handle debug events (recursion steps) if present
                if "debug" in result:
                    debug_event = result["debug"]
                    step = debug_event.get("step")
                    if isinstance(step, int):
                        logger.sync_recursion(step)
                    continue

                # Handle streaming chunks
                if "chunk" in result:
                    chunk = result["chunk"]

                    if "messages" in chunk and len(chunk["messages"]) > 0:
                        last_message = chunk["messages"][-1]
                        # Treat every new AI reasoning step as a recursion start
                        if getattr(last_message, "type", None) == "ai":
                            logger.on_recursion_start()

                        # Detect tool calls (start of tool execution)
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                tool_name = tool_call.get('name', '')
                                tool_call_id = tool_call.get('id', '')
                                tool_args = tool_call.get('args', {})
                                logger.on_tool_call(tool_name, tool_call_id, tool_args)

                        # Detect tool results
                        if hasattr(last_message, 'type') and last_message.type == 'tool':
                            tool_call_id = getattr(last_message, 'tool_call_id', None)
                            if tool_call_id:
                                # Pass token counter directly instead of parsing strings
                                display_data = logger.on_tool_result(
                                    tool_call_id,
                                    last_message.content,
                                    token_counter
                                )
                                # Update live display
                                live.update(format_display(display_data))

                # Extract the final output
                if "agent" in result:
                    agent_output = result["agent"].get("output", "")

                    # Stop live display
                    live.stop()

                    # Display final report
                    console.print("\n" + "=" * 80)
                    console.print("✎ RESEARCH REPORT", style="bold blue")
                    console.print("=" * 80 + "\n")
                    console.print(Markdown(agent_output))

                    # Save to file
                    report_dir = db_path.parent / ".bartleby"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_path = report_dir / "report.md"

                    with report_path.open("w", encoding="utf-8") as f:
                        f.write(agent_output)

                    console.print(f"\n☑ Report saved to: {report_path}")

                    # Show final stats
                    display_data = logger.get_display_data()
                    console.print(
                        f"[dim]Final: {display_data['recursion']} recursions | {display_data['token_stats_formatted']}[/dim]"
                    )
                    return

    except KeyboardInterrupt:
        send("\n\nResearch interrupted by user.", "WARN")
        display_data = logger.get_display_data()
        send(f"Token usage: {display_data['token_stats_formatted']}", "TOKENS")
        return
    except GraphRecursionError:
        display_data = logger.get_display_data()
        send(
            "Reached the recursion limit before completing the report. "
            "You can raise `max_recursions` in ~/.bartleby/config.yaml or adjust your prompt to be more specific.",
            "ERROR"
        )
        send(f"Token usage: {display_data['token_stats_formatted']}", "TOKENS")
        return
    except Exception as e:
        display_data = logger.get_display_data()
        send(f"Error during research: {e}", "ERROR")
        send(f"Token usage: {display_data['token_stats_formatted']}", "TOKENS")
        raise
