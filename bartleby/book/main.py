"""Main entry point for the book command."""

from pathlib import Path

from rich.console import Console

from bartleby.book.sessions import parse_sessions
from bartleby.book.views import render_logs, render_notes, render_overview, render_sessions


def main(
    db_path: Path,
    subcommand: str | None = None,
    full: bool = False,
    session_filter: str | None = None,
):
    """Run the book command.

    Args:
        db_path: Path to the project database (used to find book_dir)
        subcommand: One of None (overview), "sessions", "notes", "logs"
        full: For notes, show full content
        session_filter: Filter by session name/uuid
    """
    console = Console()
    book_dir = db_path.parent / "book"
    project_name = db_path.parent.name

    if not book_dir.exists():
        console.print(f"[yellow]No book directory found for project '{project_name}'.[/yellow]")
        console.print("[dim]Run 'bartleby write' to start researching and generate findings.[/dim]")
        return

    sessions = parse_sessions(book_dir)

    if subcommand is None:
        # Overview
        render_overview(console, project_name, sessions, book_dir)
    elif subcommand == "sessions":
        render_sessions(console, sessions)
    elif subcommand == "notes":
        render_notes(console, sessions, full=full, filter_name=session_filter)
    elif subcommand == "logs":
        render_logs(console, sessions, filter_session=session_filter)
    else:
        console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
