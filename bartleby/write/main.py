"""Main entry point for the write (research agent) command."""

import os
import re
import signal
import sys
import time
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path
from threading import Lock

from loguru import logger
from rich.prompt import Prompt
from rich.console import Console
from smolagents import (
    ActionStep,
    FinalAnswerStep,
    ToolCallingAgent,
    ToolCall as SmolToolCall,
)

# Disable tokenizers parallelism warning (set before importing sentence_transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import CrossEncoder, SentenceTransformer

from bartleby.lib.config import load_config
from bartleby.lib.consts import EMBEDDING_MODEL, RERANKER_MODEL
from bartleby.lib.utils import build_model_id, load_model_from_config
from bartleby.read.sqlite import get_connection
from bartleby.write.events import (
    AnswerEvent,
    BudgetWarningEvent,
    EventBus,
    ReadyEvent,
    ResearchSummaryEvent,
    SessionInfoEvent,
    SplashEvent,
    StatusEvent,
    ThinkingEvent,
    TokenStatsEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from bartleby.write.logging import StreamingLogger
from bartleby.write.progress import extract_tool_summary
from bartleby.write.references import ReferenceRegistry
from bartleby.write.renderer import CliRenderer, EventCapturingLogger
from bartleby.write.search import get_chunk_window_by_chunk_id, list_all_documents
from bartleby.write.skills import collect_tools
from bartleby.write.skills._base import slugify
from bartleby.write.token_counter import BudgetExceededError, TokenCounter
from bartleby.write.views import render_browse_view, render_sources_table


def _load_prompt(name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return ""


def _load_memory_notes(memory_dir: Path) -> str:
    """Load curated notes from memory for context injection."""
    if not memory_dir.exists():
        return ""

    note_files = sorted(memory_dir.glob("*.md"))
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


def _make_subagent_callback(bus: EventBus, token_counter: TokenCounter):
    """Create a step callback that emits subagent events.

    Timing: smolagents attaches both tool_calls and observations to the same
    ActionStep, so we measure elapsed time between consecutive callbacks
    (which approximates model-inference + tool-execution time per step).
    """
    _step_start = [time.monotonic()]

    def on_step(step):
        token_counter.on_step(step)

        now = time.monotonic()
        if not isinstance(step, ActionStep):
            return

        elapsed = now - _step_start[0]
        _step_start[0] = now

        if hasattr(step, "tool_calls") and step.tool_calls:
            obs = str(getattr(step, "observations", ""))
            tools = [
                tc for tc in step.tool_calls
                if getattr(tc, "name", "") not in ("", "final_answer")
            ]
            per_tool = elapsed / max(len(tools), 1)
            for tc in tools:
                name = getattr(tc, "name", "")
                bus.emit(ToolStartEvent(
                    tool_name=name,
                    tool_args=getattr(tc, "arguments", {}),
                    is_subagent=True,
                ))
                bus.emit(ToolCompleteEvent(
                    tool_name=name,
                    observations=obs,
                    elapsed=per_tool,
                    is_subagent=True,
                ))

    return on_step


def _build_search_subagent(model, context, event_logger, bus: EventBus, token_counter: TokenCounter):
    """Create the search subagent with auto-discovered tools."""
    search_tools = collect_tools("search_expert", context)

    # Build search prompt with corpus overview
    base_prompt = _load_prompt("search_agent")
    corpus_overview = _build_corpus_overview(context["connection"])
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
        logger=event_logger,
        step_callbacks=[_make_subagent_callback(bus, token_counter)],
    )


def _create_session_dir(book_dir: Path, run_uuid: str) -> Path:
    """Create a session directory with a date-based temporary name."""
    sessions_dir = book_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    session_dir = sessions_dir / f"{today}_{run_uuid}"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "search_reports").mkdir(exist_ok=True)
    return session_dir


def _rename_session_dir(session_dir: Path, model, questions: list[str]) -> Path:
    """Rename the session dir with a human-readable slug derived from questions."""
    if not questions:
        return session_dir

    prompt = (
        "Generate a 3-4 word kebab-case summary of these research questions. "
        "Return ONLY the slug, nothing else. Example: pm25-health-disparities\n\n"
        + "\n".join(f"- {q}" for q in questions)
    )

    try:
        result = model.generate([{"role": "user", "content": prompt}])
        slug = (result.content if hasattr(result, "content") else str(result)).strip()
        # Clean the slug
        slug = re.sub(r"[^\w-]", "", slug.lower().strip('"\'`'))
        slug = re.sub(r"-+", "-", slug).strip("-")
        if not slug:
            return session_dir
    except Exception:
        return session_dir

    # Preserve the original date prefix from the session dir name
    date_prefix = session_dir.name.split("_")[0]
    new_name = f"{date_prefix}_{slug}"
    new_dir = session_dir.parent / new_name

    # Handle collision
    if new_dir.exists() and new_dir != session_dir:
        new_dir = session_dir.parent / f"{new_name}-{session_dir.name.split('_')[-1][:4]}"

    try:
        session_dir.rename(new_dir)
        return new_dir
    except OSError:
        return session_dir


def _append_transcript(session_dir: Path, question: str, answer: str):
    """Append a Q&A pair to the session transcript."""
    transcript_path = session_dir / "transcript.md"

    is_new = not transcript_path.exists() or transcript_path.stat().st_size == 0
    with transcript_path.open("a", encoding="utf-8") as f:
        if not is_new:
            f.write("\n---\n\n")
        f.write(f"## Q: {question}\n\n{answer}\n")


def _save_search_report(session_dir: Path, task: str, observations: str, index: int):
    """Save a search agent's full synthesis to the session's search_reports/."""
    slug = slugify(task, max_len=40)
    filename = f"{index:02d}-{slug}.md"
    report_path = session_dir / "search_reports" / filename
    report_path.write_text(f"# Search: {task}\n\n{observations}\n", encoding="utf-8")


def _generate_research_summary(model, tool_log: list[dict]) -> str:
    """Generate a brief LLM summary of the research process."""
    if not tool_log:
        return ""

    steps = []
    for entry in tool_log:
        name = entry.get("name", "")
        args = entry.get("args", {})
        summary = entry.get("summary", "")
        if name == "search_expert":
            task = args.get("task", "")
            steps.append(f"- Searched: {task}")
        elif name == "get_chunk_window":
            steps.append(f"- Read passage in context ({summary})")
        elif name == "save_note":
            steps.append(f"- Saved note ({summary})")
        elif name != "final_answer":
            steps.append(f"- {name}: {summary}")

    if not steps:
        return ""

    prompt = (
        "Summarize this research process in 1-2 sentences. "
        "Be specific about what was searched and found. "
        "Do not use phrases like 'the agent' — write in passive voice.\n\n"
        + "\n".join(steps)
    )

    try:
        result = model.generate(
            [{"role": "user", "content": prompt}],
        )
        return result.content if hasattr(result, "content") else str(result)
    except Exception:
        return ""


def main(db_path: Path, verbose: bool = False):
    """Run the research agent in conversational mode."""
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Configure logging level
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    config = load_config()
    run_uuid = uuid.uuid4().hex[:8]

    # Set up directories
    book_dir = db_path.parent / "book"
    book_dir.mkdir(parents=True, exist_ok=True)

    memory_dir = db_path.parent / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    session_dir = _create_session_dir(book_dir, run_uuid)

    # Ensure reports dir exists
    (book_dir / "reports").mkdir(exist_ok=True)

    # Event architecture
    bus = EventBus()
    console = Console(force_terminal=True, force_interactive=True)
    renderer = CliRenderer(console)
    bus.subscribe(renderer.on_event)

    bus.emit(SplashEvent())

    # Validate database exists
    if not db_path.exists():
        bus.emit(StatusEvent(f"Database not found: {db_path}"))
        console.print("[dim]Please run 'bartleby read' first to process documents.[/dim]")
        return

    # Load model from config
    bus.emit(StatusEvent("Loading LLM..."))
    model = load_model_from_config(config)
    if model is None:
        console.print("[bold red]No LLM configured[/bold red]")
        console.print("[dim]Please run 'bartleby ready' first to configure your LLM.[/dim]")
        return

    # Load embedding model
    bus.emit(StatusEvent("Loading embedding model..."))
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load cross-encoder re-ranker
    reranker = None
    try:
        bus.emit(StatusEvent("Loading re-ranker..."))
        reranker = CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        logger.warning(f"Could not load re-ranker: {e}")

    # Open a single DB connection for the session
    connection = get_connection(db_path)

    # Count documents for session info
    docs = list_all_documents(connection)
    doc_count = len(docs)

    # Show session info
    project_name = config.get("active_project", "")
    model_name = config.get("model", "")
    bus.emit(SessionInfoEvent(
        project=project_name,
        model=model_name,
        doc_count=doc_count,
        session_id=run_uuid,
    ))

    # Build shared context
    token_counter = TokenCounter(model_name=model_name)

    streaming_logger = StreamingLogger(
        session_dir=session_dir,
        token_counter=token_counter,
    )

    ref_registry = ReferenceRegistry()

    # Event-capturing logger replaces verbosity_level
    event_logger = EventCapturingLogger(bus)

    agent_ref: list = []

    context = {
        "connection": connection,
        "memory_dir": memory_dir,
        "embedding_model": embedding_model,
        "embedding_lock": Lock(),
        "run_uuid": run_uuid,
        "model_id": build_model_id(config),
        "ref_registry": ref_registry,
        "reranker": reranker,
        "renderer": renderer,
        "agent_ref": agent_ref,
        "book_dir": book_dir,
    }

    # Build search subagent
    search_subagent = _build_search_subagent(model, context, event_logger, bus, token_counter)

    # Main agent tools (auto-discovered from skill.md frontmatter)
    main_tools = collect_tools("research", context)

    system_prompt = _load_prompt("agent")

    # Conversational loop
    questions_asked: list[str] = []
    search_report_index = 0
    last_answer = ""

    bus.emit(ReadyEvent())

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
                    stamp = datetime.now().strftime("%Y%m%d%H%M")
                    report_path = book_dir / "reports" / f"report-{stamp}.md"
                    report_path.write_text(last_answer, encoding="utf-8")
                    console.print(f"[green]Saved to {report_path}[/green]")
                else:
                    console.print("[yellow]No answer to save yet.[/yellow]")
                continue

            # Handle /browse command
            if stripped.lower().startswith("/browse"):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 1:
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

            # Inject curated memory notes for continuity
            notes = _load_memory_notes(memory_dir)
            if notes:
                augmented_input = f"## Previous research notes\n\n{notes}\n\n---\n\n## Current question\n\n{stripped}"
            else:
                augmented_input = stripped

            # Run agent with live progress display
            max_steps = 10
            tool_log: list[dict] = []
            renderer.start_live(max_steps, token_counter=token_counter)

            agent = ToolCallingAgent(
                tools=main_tools,
                model=model,
                managed_agents=[search_subagent],
                max_steps=max_steps,
                instructions=system_prompt,
                step_callbacks=[on_step],
                logger=event_logger,
            )
            agent_ref.clear()
            agent_ref.append(agent)

            try:
                result = None
                _active_tools: set[str] = set()
                for event in agent.run(augmented_input, stream=True):
                    if isinstance(event, SmolToolCall):
                        if event.name != "final_answer":
                            _active_tools.add(event.name)
                            bus.emit(ToolStartEvent(
                                tool_name=event.name,
                                tool_args=getattr(event, "arguments", {}),
                            ))
                    elif isinstance(event, ActionStep):
                        if hasattr(event, "tool_calls") and event.tool_calls:
                            obs = str(getattr(event, "observations", ""))
                            for tc in event.tool_calls:
                                tc_name = getattr(tc, "name", "")
                                if tc_name and tc_name != "final_answer":
                                    # Emit start if we missed the ToolCall event
                                    if tc_name not in _active_tools:
                                        bus.emit(ToolStartEvent(
                                            tool_name=tc_name,
                                            tool_args=getattr(tc, "arguments", {}),
                                        ))
                                    _active_tools.discard(tc_name)
                                    bus.emit(ToolCompleteEvent(
                                        tool_name=tc_name,
                                        observations=obs,
                                    ))
                                    tool_log.append({
                                        "name": tc_name,
                                        "args": getattr(tc, "arguments", {}),
                                        "summary": extract_tool_summary(tc_name, obs),
                                    })
                                    # Auto-save search reports
                                    if tc_name == "search_expert":
                                        search_report_index += 1
                                        task = getattr(tc, "arguments", {}).get("task", "")
                                        _save_search_report(session_dir, task, obs, search_report_index)
                            # Sync progress max_steps after step extensions
                            if renderer.progress:
                                renderer.progress.max_steps = agent.max_steps
                        else:
                            bus.emit(ThinkingEvent())
                    elif isinstance(event, FinalAnswerStep):
                        result = event.output
                    else:
                        logger.debug(f"Unhandled stream event: {type(event).__name__}")
                answer = str(result) if result else "I couldn't generate an answer."
            except BudgetExceededError:
                answer = str(result) if result else "Token budget exceeded. Here is what I found so far."
                bus.emit(BudgetWarningEvent(message="Token budget exceeded — wrapping up."))
            except Exception as e:
                if verbose:
                    answer = f"Error: {e}\n\n```\n{traceback.format_exc()}```"
                else:
                    answer = f"Error: {e}"
            finally:
                renderer.stop_live()

            # Show research summary
            research_summary = _generate_research_summary(model, tool_log)
            if research_summary:
                bus.emit(ResearchSummaryEvent(summary=research_summary))

            # Display answer
            bus.emit(AnswerEvent(text=answer))

            # Display sources table
            if ref_registry.all_refs():
                render_sources_table(console, ref_registry)

            # Show token usage
            bus.emit(TokenStatsEvent(stats=token_counter.get_stats()))

            # Save to session transcript
            _append_transcript(session_dir, stripped, answer)
            questions_asked.append(stripped)

            # Rename session dir with a human-readable slug after first question
            if len(questions_asked) == 1:
                session_dir = _rename_session_dir(session_dir, model, questions_asked)
                # Update streaming logger path
                streaming_logger.log_path = session_dir / "log.jsonl"

            last_answer = answer

    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting.[/yellow]")
    finally:
        connection.close()

    # Final statistics
    console.print(f"\nTokens: {token_counter.get_stats()}")
    console.print(f"Session: {session_dir}")
    console.print(f"Memory: {memory_dir}")
