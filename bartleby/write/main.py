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
from smolagents import (
    ActionStep,
    FinalAnswerStep,
    LogLevel,
    ToolCallingAgent,
    ToolCall as SmolToolCall,
)

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
from bartleby.write.references import ReferenceRegistry
from bartleby.write.search import get_chunk_window_by_chunk_id, list_all_documents
from bartleby.write.skills.search import HybridSearchTool, GetChunkWindowTool, GetFullDocumentTool
from bartleby.write.skills.library import (
    ListDocumentsTool,
    GetDocumentSummaryTool,
    SummarizeDocumentTool,
)
from bartleby.write.skills.memory import ReadNotesTool, SaveNoteTool
from bartleby.write.skills.file_write import WriteFileTool
from bartleby.write.token_counter import TokenCounter
from bartleby.write.views import render_browse_view, render_sources_table


def _load_prompt(name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return ""


def _load_session_notes(findings_dir: Path) -> str:
    """Load existing session notes for context injection."""
    if not findings_dir.exists():
        return ""

    note_files = sorted(findings_dir.glob("*.md"))
    if not note_files:
        return ""

    parts = []
    for note_file in note_files:
        parts.append(note_file.read_text(encoding="utf-8").strip())

    return "\n\n---\n\n".join(parts)


def _build_corpus_overview(connection) -> str:
    """Build a brief corpus overview for the search subagent's context."""
    docs = list_all_documents(connection)
    if not docs:
        return "No documents in the corpus."

    lines = [f"Corpus contains {len(docs)} document(s):\n"]
    for doc in docs:
        title = doc.get("title") or doc.get("filename") or doc.get("document_id", "unknown")
        pages = doc.get("pages_count", "?")
        lines.append(f"- {title} ({pages} pages) [id: {doc['document_id']}]")

    return "\n".join(lines)


def _build_search_subagent(model, context, log_level):
    """Create the search subagent with search/library tools."""
    connection = context["connection"]
    embedding_model = context.get("embedding_model")
    embedding_lock = context.get("embedding_lock")
    ref_registry = context.get("ref_registry")
    reranker = context.get("reranker")
    model_id = context.get("model_id")

    search_tools = [
        HybridSearchTool(
            connection,
            embedding_model=embedding_model,
            embedding_lock=embedding_lock,
            ref_registry=ref_registry,
            reranker=reranker,
        ),
        GetChunkWindowTool(connection, ref_registry=ref_registry),
        GetFullDocumentTool(connection),
        ListDocumentsTool(connection),
        GetDocumentSummaryTool(connection),
        SummarizeDocumentTool(connection, model_id),
    ]

    # Build search prompt with corpus overview
    base_prompt = _load_prompt("search_agent")
    corpus_overview = _build_corpus_overview(connection)
    search_prompt = f"{base_prompt}\n\n## Corpus\n\n{corpus_overview}"

    return ToolCallingAgent(
        tools=search_tools,
        model=model,
        name="search_expert",
        description=(
            "A document search specialist. Give it a research question or topic "
            "and it will search the corpus, read relevant passages, and return "
            "a synthesis with citation references [1], [2], etc."
        ),
        max_steps=5,
        instructions=search_prompt,
        verbosity_level=log_level,
    )


def _build_search_log_path(book_dir: Path, run_uuid: str) -> Path:
    """Get the path for the search session log."""
    return book_dir / f"searches-{run_uuid}.md"


def _log_search_invocation(log_path: Path, task: str, result: str):
    """Append a search invocation to the session log."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n---\n\n### Search at {timestamp}\n\n**Task:** {task}\n\n**Result:**\n\n{result}\n"

    is_new = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# Search Session Log\n\nSession: {log_path.stem}\n")
        f.write(entry)


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
    search_log_path = _build_search_log_path(book_dir, run_uuid)

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

    # Build shared context
    model_name = config.get("model", "")
    token_counter = TokenCounter(model_name=model_name)
    console = Console(force_terminal=True, force_interactive=True)

    streaming_logger = StreamingLogger(
        log_path=log_path,
        token_counter=token_counter,
        run_uuid=run_uuid,
    )

    ref_registry = ReferenceRegistry()

    agent_log_level = LogLevel.DEBUG if verbose else LogLevel.ERROR

    context = {
        "connection": connection,
        "findings_dir": findings_dir,
        "embedding_model": embedding_model,
        "embedding_lock": Lock(),
        "run_uuid": run_uuid,
        "model_id": build_model_id(config),
        "ref_registry": ref_registry,
    }

    # Build search subagent
    search_subagent = _build_search_subagent(model, context, agent_log_level)

    # Main agent tools: browse refs directly, notes, file writing
    main_tools = [
        GetChunkWindowTool(connection, ref_registry=ref_registry),
        ReadNotesTool(findings_dir),
        SaveNoteTool(findings_dir, run_uuid),
        WriteFileTool(findings_dir.parent),
    ]

    system_prompt = _load_prompt("agent")

    # Conversational loop
    finding_index = 0
    last_answer = ""

    console.print("[bold]Ask questions about your documents. Type /save to save the last answer as a report.[/bold]")
    console.print("[bold]Type /browse <#> to view a cited source passage in context.[/bold]")
    console.print("[dim]Press Ctrl+C to exit.[/dim]\n")

    def on_step(step):
        """Step callback for token tracking and logging."""
        token_counter.on_step(step)

        if not isinstance(step, ActionStep):
            return

        if hasattr(step, "tool_calls") and step.tool_calls:
            for tool_call in step.tool_calls:
                tool_name = getattr(tool_call, "name", "")
                tool_id = getattr(tool_call, "id", str(id(tool_call)))
                tool_args = getattr(tool_call, "arguments", {})
                streaming_logger.on_tool_call(tool_name, tool_id, tool_args)

                # Log search subagent invocations
                if tool_name == "search_expert":
                    task = tool_args.get("task", "")
                    if task:
                        _log_search_invocation(search_log_path, task, "(pending)")

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

            # Handle /browse command
            if stripped.lower().startswith("/browse"):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 1:
                    # /browse with no arg — re-display sources table
                    if ref_registry.all_refs():
                        render_sources_table(console, ref_registry)
                    else:
                        console.print("[yellow]No sources yet. Ask a question first.[/yellow]")
                    continue

                arg = parts[1].strip()
                try:
                    ref_num = int(arg)
                except ValueError:
                    console.print(f"[red]Invalid ref number: {arg}[/red]")
                    continue

                ref_entry = ref_registry.get(ref_num)
                if ref_entry is None:
                    total = len(ref_registry.all_refs())
                    if total == 0:
                        console.print("[yellow]No sources yet. Ask a question first.[/yellow]")
                    else:
                        console.print(f"[red]Ref [{ref_num}] not found. Valid range: 1-{total}[/red]")
                    continue

                window_data = get_chunk_window_by_chunk_id(
                    connection, ref_entry["chunk_id"], window_radius=3,
                )
                if window_data is None:
                    console.print(f"[red]Chunk not found in database for ref [{ref_num}].[/red]")
                    continue

                render_browse_view(console, ref_entry, window_data)
                continue

            # Clear refs for new question
            ref_registry.clear()

            # Inject previous session notes for continuity
            notes = _load_session_notes(findings_dir)
            if notes:
                augmented_input = f"## Previous research notes\n\n{notes}\n\n---\n\n## Current question\n\n{stripped}"
            else:
                augmented_input = stripped

            # Run agent with live spinner status
            progress = ProgressDisplay()
            with Live(progress, console=console, refresh_per_second=4):
                agent = ToolCallingAgent(
                    tools=main_tools,
                    model=model,
                    managed_agents=[search_subagent],
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

            # Display sources table
            if ref_registry.all_refs():
                render_sources_table(console, ref_registry)

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
