"""Rich terminal UI views for bartleby book."""

from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from bartleby.book.sessions import Session, Note, get_reports, parse_memory_notes


def _format_time_ago(dt: datetime | None) -> str:
    """Format a datetime as a human-readable time ago string."""
    if dt is None:
        return "unknown"

    now = datetime.now()
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds // 60)
        return f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds // 86400)
        return f"{days}d ago" if days > 1 else "yesterday"
    else:
        return dt.strftime("%Y-%m-%d")


def _format_tokens(count: int) -> str:
    """Format token count with k suffix for thousands."""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def render_overview(console: Console, project_name: str, sessions: list[Session], book_dir: Path, memory_dir: Path | None = None):
    """Render the overview panel for `bartleby book`."""
    memory_notes = parse_memory_notes(memory_dir) if memory_dir else []
    total_input = sum(s.total_input_tokens for s in sessions)
    total_output = sum(s.total_output_tokens for s in sessions)
    reports = get_reports(book_dir)

    lines = [
        f"Sessions: [bold]{len(sessions)}[/bold]",
        f"Notes: [bold]{len(memory_notes)}[/bold]",
        f"Reports: [bold]{len(reports)}[/bold]",
        f"Total tokens: [dim]{_format_tokens(total_input)} in / {_format_tokens(total_output)} out[/dim]",
    ]

    if sessions:
        most_recent = sessions[0]
        lines.append("")
        lines.append(f"[dim]Last session:[/dim] {most_recent.session_name} ({_format_time_ago(most_recent.start_time)})")

    panel = Panel(
        "\n".join(lines),
        title=f"[bold]{project_name}[/bold]",
        border_style="blue",
    )
    console.print(panel)


def render_sessions(console: Console, sessions: list[Session]):
    """Render the sessions table for `bartleby book sessions`."""
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions", show_header=True, header_style="bold")
    table.add_column("Session", style="cyan")
    table.add_column("Time", style="dim")
    table.add_column("Tools", justify="right")
    table.add_column("Tokens", justify="right", style="dim")

    for session in sessions:
        time_str = _format_time_ago(session.start_time)
        tools = str(len(session.tool_calls))
        tokens = _format_tokens(session.total_input_tokens + session.total_output_tokens)

        table.add_row(session.session_name, time_str, tools, tokens)

    console.print(table)


def render_notes(console: Console, sessions: list[Session], memory_dir: Path | None = None, full: bool = False, filter_name: str | None = None):
    """Render the notes view for `bartleby book notes`."""
    # Load notes from memory directory
    notes = parse_memory_notes(memory_dir) if memory_dir else []

    # Fall back to session-embedded notes (legacy)
    if not notes:
        for session in sessions:
            notes.extend(session.notes)

    # Sort by timestamp, most recent first
    notes.sort(key=lambda n: n.timestamp or datetime.min, reverse=True)

    if filter_name:
        filter_lower = filter_name.lower()
        notes = [n for n in notes if filter_lower in n.filename.lower() or filter_lower in n.title.lower()]

    if not notes:
        console.print("[dim]No notes found.[/dim]")
        return

    if full:
        for note in notes:
            time_str = _format_time_ago(note.timestamp)
            console.print()
            console.print(Panel(
                Markdown(note.content),
                title=f"[bold]{note.title}[/bold]",
                subtitle=f"[dim]{note.filename} \u00b7 {time_str}[/dim]",
                border_style="green",
            ))
    else:
        lines = []
        for note in notes:
            time_str = _format_time_ago(note.timestamp)
            lines.append(f"\u2022 [bold]{note.title}[/bold]")
            lines.append(f"  [dim]{note.filename} \u00b7 {time_str}[/dim]")
            lines.append("")

        lines.append("[dim]Use --full to see content[/dim]")

        panel = Panel(
            "\n".join(lines),
            title=f"[bold]Notes ({len(notes)})[/bold]",
            border_style="green",
        )
        console.print(panel)


def render_logs(console: Console, sessions: list[Session], filter_session: str | None = None):
    """Render the logs view for `bartleby book logs`."""
    if filter_session:
        filter_lower = filter_session.lower()
        sessions = [
            s for s in sessions
            if filter_lower in s.session_name.lower()
        ]

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    # Aggregate tool call stats
    tool_stats: Counter[str] = Counter()

    for session in sessions:
        for tc in session.tool_calls:
            tool_stats[tc.tool_name] += 1

    # Build summary table
    table = Table(title="Tool Usage", show_header=True, header_style="bold")
    table.add_column("Tool", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("% of total", justify="right", style="dim")

    total_calls = sum(tool_stats.values())
    for tool_name, count in tool_stats.most_common():
        pct = f"{100 * count / total_calls:.1f}%" if total_calls > 0 else "0%"
        table.add_row(tool_name, str(count), pct)

    console.print(table)

    # Token summary
    total_input = sum(s.total_input_tokens for s in sessions)
    total_output = sum(s.total_output_tokens for s in sessions)

    console.print()
    console.print(f"[bold]Total tokens:[/bold] {_format_tokens(total_input)} input, {_format_tokens(total_output)} output")

    if filter_session and len(sessions) == 1:
        session = sessions[0]
        console.print()
        console.print(f"[bold]Session:[/bold] {session.session_name}")
        console.print()

        detail_table = Table(show_header=True, header_style="bold")
        detail_table.add_column("Time", style="dim")
        detail_table.add_column("Tool", style="cyan")
        detail_table.add_column("Tokens", justify="right", style="dim")

        for tc in session.tool_calls:
            time_str = tc.timestamp.strftime("%H:%M:%S")
            tokens = f"{_format_tokens(tc.input_tokens)}/{_format_tokens(tc.output_tokens)}"
            detail_table.add_row(time_str, tc.tool_name, tokens)

        console.print(detail_table)
