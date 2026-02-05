"""Rich terminal UI views for bartleby book."""

from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bartleby.book.sessions import Session, Note, get_reports, short_cute_name


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


def render_overview(console: Console, project_name: str, sessions: list[Session], book_dir: Path):
    """Render the overview panel for `bartleby book`."""
    total_notes = sum(len(s.notes) for s in sessions)
    total_input = sum(s.total_input_tokens for s in sessions)
    total_output = sum(s.total_output_tokens for s in sessions)
    reports = get_reports(book_dir)

    lines = [
        f"Sessions: [bold]{len(sessions)}[/bold]",
        f"Notes: [bold]{total_notes}[/bold]",
        f"Reports: [bold]{len(reports)}[/bold]",
        f"Total tokens: [dim]{_format_tokens(total_input)} in / {_format_tokens(total_output)} out[/dim]",
    ]

    if sessions:
        most_recent = sessions[0]
        lines.append("")
        lines.append(f"[dim]Last session:[/dim] {short_cute_name(most_recent.run_uuid)} ({_format_time_ago(most_recent.start_time)})")

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
    table.add_column("Notes", justify="right")
    table.add_column("Tokens", justify="right", style="dim")

    for session in sessions:
        name = short_cute_name(session.run_uuid)
        time_str = _format_time_ago(session.start_time)
        tools = str(len(session.tool_calls))
        notes = str(len(session.notes))
        tokens = _format_tokens(session.total_input_tokens + session.total_output_tokens)

        table.add_row(name, time_str, tools, notes, tokens)

    console.print(table)


def render_notes(console: Console, sessions: list[Session], full: bool = False, filter_name: str | None = None):
    """Render the notes view for `bartleby book notes`."""
    # Collect all notes across sessions
    all_notes: list[tuple[Note, Session]] = []
    for session in sessions:
        for note in session.notes:
            all_notes.append((note, session))

    # Sort by timestamp, most recent first
    all_notes.sort(key=lambda x: x[0].timestamp or datetime.min, reverse=True)

    if filter_name:
        # Filter to a specific session by cute name prefix
        all_notes = [
            (n, s) for n, s in all_notes
            if filter_name.lower() in short_cute_name(s.run_uuid).lower()
            or filter_name.lower() in s.run_uuid.lower()
        ]

    if not all_notes:
        console.print("[dim]No notes found.[/dim]")
        return

    if full:
        # Show full content
        for note, session in all_notes:
            session_name = short_cute_name(session.run_uuid)
            time_str = _format_time_ago(note.timestamp)

            console.print()
            console.print(Panel(
                Markdown(note.content),
                title=f"[bold]{note.title}[/bold]",
                subtitle=f"[dim]{session_name} \u00b7 {time_str}[/dim]",
                border_style="green" if note.is_explicit_note else "blue",
            ))
    else:
        # Show titles only
        lines = []
        for note, session in all_notes:
            session_name = short_cute_name(session.run_uuid)
            time_str = _format_time_ago(note.timestamp)
            icon = "\u2022" if note.is_explicit_note else "\u25e6"
            lines.append(f"{icon} [bold]{note.title}[/bold]")
            lines.append(f"  [dim]{session_name} \u00b7 {time_str}[/dim]")
            lines.append("")

        lines.append("[dim]Use --full or bartleby book notes <session> to see content[/dim]")

        panel = Panel(
            "\n".join(lines),
            title=f"[bold]Notes ({len(all_notes)})[/bold]",
            border_style="green",
        )
        console.print(panel)


def render_logs(console: Console, sessions: list[Session], filter_session: str | None = None):
    """Render the logs view for `bartleby book logs`."""
    # Filter sessions if requested
    if filter_session:
        sessions = [
            s for s in sessions
            if filter_session.lower() in short_cute_name(s.run_uuid).lower()
            or filter_session.lower() in s.run_uuid.lower()
        ]

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    # Aggregate tool call stats
    tool_stats: Counter[str] = Counter()
    tool_times: dict[str, list[float]] = {}

    for session in sessions:
        prev_time = None
        for tc in session.tool_calls:
            tool_stats[tc.tool_name] += 1

            # Estimate duration from timestamps (time until next call or end)
            if prev_time is not None:
                duration = (tc.timestamp - prev_time).total_seconds()
                if tc.tool_name not in tool_times:
                    tool_times[tc.tool_name] = []
                # Use previous tool's duration
            prev_time = tc.timestamp

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
        # Show detailed log for single session
        session = sessions[0]
        console.print()
        console.print(f"[bold]Session:[/bold] {short_cute_name(session.run_uuid)}")
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
