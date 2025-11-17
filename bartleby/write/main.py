"""Main entry point for the write (research agent) command."""

import os
import time
import uuid
from pathlib import Path

from rich.prompt import Prompt
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.spinner import Spinner
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
    DEFAULT_MAX_SEARCH_OPERATIONS,
    DEFAULT_MAX_TODO_ROUNDS,
    DEFAULT_MAX_TOTAL_ROUNDS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_QA_ENABLED,
    DEFAULT_QA_SAVE_SESSION,
    EMBEDDING_MODEL,
)
from bartleby.lib.utils import load_config, load_llm_from_config
from bartleby.write.primary_agent import run_primary_agent
from bartleby.write.logging import StreamingLogger
from bartleby.write.token_counter import TokenCounterCallback


def format_qa_display(display_data: dict) -> Text:
    """
    Render Q&A session display data.

    Args:
        display_data: Dict with Q&A-specific display data

    Returns:
        Rich Text object for Live display
    """
    def _wrap(text: Text) -> Text:
        text.no_wrap = False
        text.overflow = "fold"
        return text

    lines: list[Text] = []

    # Current question (if active)
    if display_data.get('current_question'):
        question_line = Text.assemble(
            (" ∟ ", "dim"),
            ("Q: ", "bold"),
            (display_data['current_question'], "white"),
        )
        lines.append(_wrap(question_line))

    # Show if Search Agent is active
    max_searches = display_data.get('max_searches', 0)
    if max_searches:
        search_count = display_data.get('search_count', 0)
        search_text = f" ∟ Search Agent: search {search_count}/{max_searches}"
        lines.append(_wrap(Text(search_text, style="yellow")))

    # Recent actions (if any)
    if display_data.get('action_history'):
        lines.append(_wrap(Text("")))
        lines.append(_wrap(Text("Recent activity:", style="dim")))
        for action in display_data['action_history']:
            lines.append(_wrap(Text(f"- {action.get('friendly_name', 'Action')}", style="dim")))

    # Current action
    if display_data.get('current_action') and display_data['current_action'].get('friendly_name'):
        if not display_data.get('action_history'):
            lines.append(_wrap(Text("")))
        lines.append(_wrap(Text(f"- {display_data['current_action']['friendly_name']}", style="bold")))

    # Token usage and spinner
    lines.append(_wrap(Text("")))
    token_text = display_data.get('token_stats_formatted', '[↑0/↓0/+0]')
    lines.append(_wrap(Text(f"Tokens: {token_text}", style="white")))
    if display_data.get('spinner_active'):
        lines.append(Spinner("dots", text="Thinking..."))
    lines.append(_wrap(Text("")))

    return Group(*lines)


def _normalize_qa_question(raw: str) -> tuple[str, bool]:
    """Return cleaned question and whether user forced delegation."""
    if raw is None:
        return "", False

    question = raw.strip()
    forced = False

    prefixes = ("/search", "/research")
    lowered = question.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            forced = True
            question = question[len(prefix):].lstrip(": ").strip()
            lowered = question.lower()
            break

    suffixes = (" (research)", " [research]", " (search)", " [search]")
    for suffix in suffixes:
        if lowered.endswith(suffix):
            forced = True
            question = question[:-len(suffix)].rstrip()
            lowered = question.lower()
            break

    return (question if question else raw.strip(), forced)


def format_display(display_data: dict) -> Text:
    """
    Render display data as plain text (no panel border).

    Args:
        display_data: Dict with agent_name, round, search_count, action_history, current_action, token_stats_formatted

    Returns:
        Rich Text object for Live display
    """
    # Check if in Q&A mode
    if display_data.get('mode') == 'qa':
        return format_qa_display(display_data)

    def _wrap(text: Text) -> Text:
        text.no_wrap = False
        text.overflow = "fold"
        return text

    lines: list[Text] = []

    round_text = f"Primary Agent: round {display_data['round']}/{display_data['max_rounds']}"
    lines.append(_wrap(Text(round_text, style="bold")))

    # Nested Search Agent status when active
    max_searches = display_data.get('max_searches', 0)
    if max_searches:
        search_count = display_data.get('search_count', 0)
        search_text = f" ∟ Search Agent: round {search_count}/{max_searches}"
        lines.append(_wrap(Text(search_text, style="cyan")))

    # Todo list
    todos = display_data.get('todos', [])
    if todos:
        lines.append(_wrap(Text("")))
        lines.append(_wrap(Text("To-do list:", style="bold")))
        for todo in todos:
            symbol = "☐"
            style = "white"
            if todo['status'] == 'complete':
                symbol = "☑"
                style = "dim"
            elif todo['status'] == 'active':
                symbol = "☐"
                style = "bold cyan"

            lines.append(_wrap(Text(f"{symbol} {todo['task']}", style=style)))

    # Actions list
    has_actions = display_data['action_history'] or display_data['current_action']['summary']
    if has_actions:
        lines.append(_wrap(Text("")))
        lines.append(_wrap(Text("Actions list:", style="bold")))

        # Action history (dim)
        for action in display_data['action_history']:
            lines.append(_wrap(Text(
                f"- {action['friendly_name']}",
                style="dim"
            )))

        # Current action (bold)
        current = display_data['current_action']
        if current['summary']:
            lines.append(_wrap(Text(
                f"- {current['friendly_name']}",
                style="bold"
            )))

    # Token stats
    lines.append(_wrap(Text("")))
    token_text = display_data['token_stats_formatted']
    lines.append(_wrap(Text(f"Tokens: {token_text}", style="white")))
    lines.append(_wrap(Text("")))

    return Group(*lines)


def main(db_path: Path, qa_only: bool = False):
    """
    Run the research agent.

    Args:
        db_path: Path to the document database
        qa_only: Skip report generation and go directly to Q&A mode
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

    # Clean up previous run files unless we're only running Q&A
    log_path = bartleby_dir_path / "log.json"
    todos_path = bartleby_dir_path / "todos.json"
    if not qa_only:
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

    # ============================================
    # Q&A Only Mode
    # ============================================
    if qa_only:
        send("Starting Q&A mode (skipping report generation)", "BIG")

        # Check for required artifacts
        report_dir = db_path.parent / ".bartleby"
        report_path = report_dir / "report.md"
        log_path = report_dir / "log.json"
        findings_dir = report_dir / "findings"

        if not report_path.exists():
            send(f"Report not found: {report_path}", "ERROR")
            send("Run 'bartleby write --db <path>' first to generate a report.", "WARN")
            return

        if not findings_dir.exists() or not list(findings_dir.glob("*.md")):
            send(f"Findings not found in: {findings_dir}", "ERROR")
            send("Run 'bartleby write --db <path>' first to generate research findings.", "WARN")
            return

        # Load existing report
        final_report_text = report_path.read_text(encoding="utf-8")
        send(f"Loaded report from: {report_path}", "COMPLETE")

        # Extract run_uuid from findings filenames (e.g., "a1b2c3d4-01.md")
        findings_files = list(findings_dir.glob("*-[0-9][0-9].md"))
        if findings_files:
            # Get uuid from first findings file
            run_uuid = findings_files[0].stem.rsplit("-", 1)[0]
        else:
            # Generate new uuid if no findings files found
            run_uuid = uuid.uuid4().hex[:8]

        # Create token counter
        model_name = config.get("model", "unknown")
        token_counter = TokenCounterCallback(model_name=model_name)
        token_counter.max_recursions = 999  # No limit for Q&A
        token_counter.token_budget = None  # No budget for Q&A

        # Load max_search_operations for Q&A delegation
        max_search_operations = _coerce_positive_int(
            config.get("max_search_operations"), DEFAULT_MAX_SEARCH_OPERATIONS
        )

        # Create logger (for Q&A logging only)
        console = Console(force_terminal=True, force_interactive=True)
        logger = StreamingLogger(
            console=console,
            llm=llm,
            log_path=log_path,
            max_recursions=999,
            todo_list=None,
            token_callback=token_counter,
            agent_name="Answer Agent",
            run_uuid=run_uuid,
        )

        # Jump to Q&A section (defined later in the file)
        # We'll use a goto-like pattern by setting a flag
        goto_qa = True
    else:
        goto_qa = False

    if not goto_qa:
        # Ask user for direction
        send("What would you like me to investigate?", "BIG")
        user_direction = Prompt.ask("Investigation direction")

        if not user_direction or not user_direction.strip():
            send("No direction provided. Exiting.", "WARN")
            return

        send("\n", "EMPTY")

    # ============================================
    # Report Generation (skip if qa_only)
    # ============================================
    if not goto_qa:
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

    # ============================================
    # Q&A Session
    # ============================================

    enable_qa = config.get("enable_qa", DEFAULT_QA_ENABLED)
    save_qa_session_enabled = config.get("qa_save_session", DEFAULT_QA_SAVE_SESSION)

    if enable_qa:
        from bartleby.write.answer_agent import initialize_answer_agent, run_answer_agent, save_qa_session

        console.print("\n" + "─" * 80)
        console.print("[bold cyan]Entering Q&A Mode[/bold cyan]")
        console.print("Ask questions about the research. Press Ctrl+C to exit.\n")

        # Initialize Answer Agent context
        answer_context = initialize_answer_agent(
            report_text=final_report_text,
            findings_dir=findings_dir,
            run_uuid=run_uuid,
            logger=logger,
        )

        console.print(f"[dim]Loaded: {answer_context['context_summary']}[/dim]\n")

        # Q&A loop
        qa_sequence = {"count": 0}
        qa_history = []
        answer_session_uuid = uuid.uuid4().hex[:8]
        answer_index = 0

        # Create Live display for Q&A mode (always start a fresh display here)
        initial_qa_display = format_display(logger.get_display_data())
        qa_live = Live(initial_qa_display, console=console, refresh_per_second=4, auto_refresh=False)
        qa_live.start()

        def render_display(data: dict):
            """Update the live display."""
            rendered = format_display(data)
            qa_live.update(rendered)
            qa_live.refresh()

        # Update display callback
        def update_display():
            """Callback for Search Agent to update display."""
            render_display(logger.get_display_data())

        try:
            while True:
                # Prompt for question
                raw_question = Prompt.ask("[bold]Question[/bold]")
                question, force_delegate = _normalize_qa_question(raw_question)

                if not question:
                    continue

                if force_delegate:
                    console.print("[dim]Running delegated research before answering...[/dim]")

                # Log question for transcripts/metrics
                logger.on_qa_question(question)

                console.print("")  # spacer before answer

                # Generate answer with live updates (show spinner in display)
                logger.qa_spinner_active = True
                render_display(logger.get_display_data())
                result = run_answer_agent(
                    question=question,
                    context=answer_context,
                    llm=llm,
                    db_path=db_path,
                    findings_dir=findings_dir,
                    embedding_model=embedding_model,
                    token_counter=token_counter,
                    logger=logger,
                    display_callback=update_display,
                    sequence_counter=qa_sequence,
                    max_search_operations=max_search_operations,
                    force_delegate=force_delegate,
                )
                logger.qa_spinner_active = False
                render_display(logger.get_display_data())

                answer = result["answer"]

                # Log answer
                logger.on_qa_answer(answer, delegated=result["delegated"])

                # Display answer
                console.print("\n[bold cyan]Answer:[/bold cyan]\n")
                console.print(Markdown(answer))
                console.print("")

                if result["delegated"]:
                    console.print("[dim]Note: Answer required additional research[/dim]\n")

                # Persist answer to markdown file
                answer_index += 1
                answer_filename = report_dir / f"answer-{answer_session_uuid}-{answer_index:02d}.md"
                answer_filename.write_text(answer, encoding="utf-8")
                console.print(f"[dim]Saved answer to {answer_filename}[/dim]")

                # Store Q&A pair
                qa_history.append({
                    "question": question,
                    "answer": answer,
                    "delegated": result["delegated"],
                    "sequence": len(qa_history) + 1
                })

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting Q&A mode[/yellow]")

        finally:
            # Stop Live display
            qa_live.stop()

        # Save Q&A session
        if qa_history and save_qa_session_enabled:
            qa_file = report_dir / "qa-session.md"
            save_qa_session(qa_history, qa_file)
            console.print(f"[dim]Q&A session saved to {qa_file}[/dim]")

        # Final statistics
        console.print("\n" + "─" * 80)
        console.print("[bold]Session Complete[/bold]")
        if token_counter:
            console.print(f"Total tokens: {token_counter.get_stats()}")
        console.print(f"Questions answered: {len(qa_history)}")
        console.print(f"Log saved to: {log_path}")
