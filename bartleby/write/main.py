"""Main entry point for the write (research agent) command."""

import os
import sys
import traceback
import uuid
from pathlib import Path
from threading import Lock

from loguru import logger
from rich.prompt import Prompt
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from smolagents import ActionStep, FinalAnswerStep, ToolCallingAgent, ToolCall as SmolToolCall, LogLevel

# Disable tokenizers parallelism warning (set before importing sentence_transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

from bartleby.lib.config import load_config
from bartleby.lib.console import send
from bartleby.lib.consts import EMBEDDING_MODEL
from bartleby.lib.utils import build_model_id, load_model_from_config
from bartleby.read.sqlite import get_connection
from bartleby.write.logging import StreamingLogger
from bartleby.write.progress import ProgressDisplay
from bartleby.write.skills import load_skills
from bartleby.write.token_counter import TokenCounter


def _load_agent_prompt() -> str:
    """Load the conversational agent system prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "agent.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "You are a research assistant with access to a document database. Search documents to answer questions."


def _load_session_notes(findings_dir: Path, run_uuid: str) -> str:
    """Load existing session notes for context injection."""
    if not findings_dir.exists():
        return ""

    pattern = f"{run_uuid}-*.md"
    note_files = sorted(findings_dir.glob(pattern))
    if not note_files:
        return ""

    parts = []
    for note_file in note_files:
        parts.append(note_file.read_text(encoding="utf-8").strip())

    return "\n\n---\n\n".join(parts)


def main(db_path: Path, verbose: bool = False):
    """Run the research agent in conversational mode."""
    # Configure logging level
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    config = load_config()
    run_uuid = uuid.uuid4().hex[:8]

    # Set up the book folder for write artifacts
    book_dir = db_path.parent / "book"
    book_dir.mkdir(parents=True, exist_ok=True)

    findings_dir = book_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    log_path = book_dir / "log.json"
    if log_path.exists():
        log_path.unlink()

    send(message_type="SPLASH")
    send("Starting Bartleby research agent", "BIG")

    # Validate database exists
    if not db_path.exists():
        send(f"Database not found: {db_path}", "ERROR")
        send("Please run 'bartleby read' first to process documents.", "WARN")
        return

    # Load model from config
    send("Loading LLM from configuration...", "BIG")
    model = load_model_from_config(config)
    if model is None:
        send("No LLM configured", "ERROR")
        send("Please run 'bartleby ready' first to configure your LLM.", "WARN")
        return

    # Load embedding model
    send(f"Loading embedding model: {EMBEDDING_MODEL}", "BIG")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Open a single DB connection for the session
    connection = get_connection(db_path)

    # Build tools via skill system
    model_name = config.get("model", "")
    token_counter = TokenCounter(model_name=model_name)
    console = Console(force_terminal=True, force_interactive=True)

    streaming_logger = StreamingLogger(
        log_path=log_path,
        token_counter=token_counter,
        run_uuid=run_uuid,
    )

    context = {
        "connection": connection,
        "findings_dir": findings_dir,
        "embedding_model": embedding_model,
        "embedding_lock": Lock(),
        "run_uuid": run_uuid,
        "model_id": build_model_id(config),
    }
    all_tools = load_skills(context)
    system_prompt = _load_agent_prompt()

    # Conversational loop
    finding_index = 0
    last_answer = ""

    console.print("[bold]Ask questions about your documents. Type /save to save the last answer as a report.[/bold]")
    console.print("[dim]Press Ctrl+C to exit.[/dim]\n")

    def on_step(step):
        """Step callback for token tracking and logging."""
        if token_counter:
            token_counter.on_step(step)

        if not isinstance(step, ActionStep):
            return

        if hasattr(step, "tool_calls") and step.tool_calls:
            for tool_call in step.tool_calls:
                tool_name = getattr(tool_call, "name", "")
                tool_id = getattr(tool_call, "id", str(id(tool_call)))
                tool_args = getattr(tool_call, "arguments", {})
                streaming_logger.on_tool_call(tool_name, tool_id, tool_args)

        if hasattr(step, "observations") and step.observations:
            pending_ids = list(streaming_logger.pending_tools.keys())
            if pending_ids:
                tool_id = pending_ids[-1]
                streaming_logger.on_tool_result(
                    tool_id,
                    str(step.observations)[:500],
                    token_counter,
                )

    try:
        while True:
            raw = Prompt.ask("[bold]>[/bold]")

            if not raw or not raw.strip():
                continue

            stripped = raw.strip()

            # Handle /save command
            if stripped.lower() == "/save":
                if last_answer:
                    from datetime import datetime
                    stamp = datetime.now().strftime("%Y%m%d%H%M")
                    report_path = book_dir / f"report-{stamp}.md"
                    report_path.write_text(last_answer, encoding="utf-8")
                    console.print(f"[green]Saved to {report_path}[/green]")
                else:
                    console.print("[yellow]No answer to save yet.[/yellow]")
                continue

            # Inject previous session notes for continuity
            notes = _load_session_notes(findings_dir, run_uuid)
            if notes:
                augmented_input = f"## Previous research notes\n\n{notes}\n\n---\n\n## Current question\n\n{stripped}"
            else:
                augmented_input = stripped

            # Run agent with live spinner status
            agent_log_level = LogLevel.DEBUG if verbose else LogLevel.ERROR
            progress = ProgressDisplay()
            with Live(progress, console=console, refresh_per_second=4):
                agent = ToolCallingAgent(
                    tools=all_tools,
                    model=model,
                    max_steps=10,
                    instructions=system_prompt,
                    step_callbacks=[on_step],
                    verbosity_level=agent_log_level,
                )

                try:
                    result = None
                    for event in agent.run(augmented_input, stream=True):
                        if isinstance(event, SmolToolCall):
                            if event.name != "final_answer":
                                progress.start_tool(event.name)
                        elif isinstance(event, ActionStep):
                            if hasattr(event, "tool_calls") and event.tool_calls:
                                obs = str(getattr(event, "observations", "")) if hasattr(event, "observations") else ""
                                for tc in event.tool_calls:
                                    tc_name = getattr(tc, "name", "")
                                    if tc_name != "final_answer":
                                        progress.complete_tool(tc_name, obs)
                            else:
                                progress.spinner.text = "Thinking..."
                        elif isinstance(event, FinalAnswerStep):
                            result = event.output
                    answer = str(result) if result else "I couldn't generate an answer."
                except Exception as e:
                    if verbose:
                        answer = f"Error: {e}\n\n```\n{traceback.format_exc()}```"
                    else:
                        answer = f"Error: {e}"

            # Display answer
            console.print("")
            console.print(Markdown(answer))
            console.print("")

            # Show token usage
            console.print(f"[dim]{token_counter.get_stats()}[/dim]\n")

            # Auto-save as finding
            finding_index += 1
            finding_file = findings_dir / f"{run_uuid}-{finding_index:02d}.md"
            finding_file.write_text(f"# Q: {stripped}\n\n{answer}\n", encoding="utf-8")

            last_answer = answer

    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting.[/yellow]")
    finally:
        connection.close()

    # Final statistics
    console.print(f"\nTokens: {token_counter.get_stats()}")
    console.print(f"Findings saved to: {findings_dir}")
    console.print(f"Log: {log_path}")
