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
    DEFAULT_MAX_TOTAL_TOKENS,
    EMBEDDING_MODEL,
)
from bartleby.lib.utils import load_config, load_llm_from_config
from bartleby.write.primary_agent import run_primary_agent, MAX_TODO_ROUNDS, MAX_TOTAL_ROUNDS
from bartleby.write.logging import StreamingLogger
from bartleby.write.token_counter import TokenCounterCallback
from bartleby.write.tools import DocumentSearchTools


def format_display(display_data: dict) -> Text:
    """
    Render display data as plain text (no panel border).

    Args:
        display_data: Dict with agent_name, round, search_count, action_history, current_action, token_stats_formatted

    Returns:
        Rich Text object for Live display
    """
    lines: list[Text] = []

    agent_name = display_data.get('agent_name', 'Primary Agent')
    round_text = f"{agent_name}: round {display_data['round']}/{display_data['max_rounds']}"
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

    # Action history (dim)
    if display_data['action_history']:
        lines.append(Text(""))
        for action in display_data['action_history']:
            lines.append(Text(
                f"{action['friendly_name']}: {action['summary']}",
                style="dim"
            ))

    # Current action
    current = display_data['current_action']
    if current['summary']:
        lines.append(Text(
            f"{current['friendly_name']}: {current['summary']}",
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

    # Initialize shared tools/resources
    model_name = config.get("model", "")
    search_tools = DocumentSearchTools(db_path, scratchpad_path, todos_path, embedding_model)

    # Load limits with safe defaults
    max_todo_rounds = _coerce_positive_int(
        config.get("max_todo_rounds"), MAX_TODO_ROUNDS
    )
    max_total_rounds = _coerce_positive_int(
        config.get("max_total_rounds"), MAX_TOTAL_ROUNDS
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

    # Logging setup
    log_path = bartleby_dir_path / "log.json"
    logger = StreamingLogger(
        console,
        llm,
        log_path,
        max_recursions=max_total_rounds,
        search_tools=search_tools,
        token_callback=token_counter,
        agent_name="Primary Agent",
    )

    final_report_text = ""

    def _run_agent():
        """Run the Primary Agent and return final report."""
        nonlocal final_report_text

        with Live(format_display(logger.get_display_data()), console=console, refresh_per_second=4) as live:
            # Define display update callback
            def update_display():
                live.update(format_display(logger.get_display_data()))

            for result in run_primary_agent(
                user_direction=user_direction,
                llm=llm,
                search_tools=search_tools,
                token_counter=token_counter,
                max_todo_rounds=max_todo_rounds,
                max_total_rounds=max_total_rounds,
                logger=logger,
                display_callback=update_display,
            ):
                if "chunk" in result:
                    chunk = result["chunk"]

                    if "messages" in chunk and len(chunk["messages"]) > 0:
                        last_message = chunk["messages"][-1]
                        if getattr(last_message, "type", None) == "ai":
                            logger.on_ai_message(last_message)

                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                tool_name = tool_call.get('name', '')
                                tool_call_id = tool_call.get('id', '')
                                tool_args = tool_call.get('args', {})
                                logger.on_tool_call(tool_name, tool_call_id, tool_args)

                        if hasattr(last_message, 'type') and last_message.type == 'tool':
                            tool_call_id = getattr(last_message, 'tool_call_id', None)
                            if tool_call_id:
                                display_data = logger.on_tool_result(
                                    tool_call_id,
                                    last_message.content,
                                    token_counter
                                )
                                live.update(format_display(display_data))

                if "agent" in result:
                    final_report_text = result["agent"].get("output", "")
                    live.stop()
                    break

    try:
        send(f"Starting Primary Agent (max {max_total_rounds} rounds)", "BIG")
        _run_agent()

        aggregate_stats = token_counter.get_stats()

    except KeyboardInterrupt:
        send("\n\nResearch interrupted by user.", "WARN")
        send(f"Token usage: {token_counter.get_stats()}", "TOKENS")
        return
    except GraphRecursionError:
        send("Agent reached the recursion limit before completing the report.", "ERROR")
        send(f"Token usage: {token_counter.get_stats()}", "TOKENS")
        return
    except Exception as e:
        send(f"Error during research: {e}", "ERROR")
        send(f"Token usage: {token_counter.get_stats()}", "TOKENS")
        raise

    if not final_report_text:
        send("Agent did not produce a final report.", "ERROR")
        send(f"Token usage: {aggregate_stats}", "TOKENS")
        return

    console.print("\n" + "=" * 80)
    console.print("✎ RESEARCH REPORT", style="bold blue")
    console.print("=" * 80 + "\n")
    console.print(Markdown(final_report_text))

    report_dir = db_path.parent / ".bartleby"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    with report_path.open("w", encoding="utf-8") as f:
        f.write(final_report_text)

    console.print(f"\n☑ Report saved to: {report_path}")

    console.print(f"[dim]Final: {logger.recursion} rounds | {aggregate_stats}[/dim]")
