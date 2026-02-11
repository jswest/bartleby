import argparse
import sys
from pathlib import Path

from loguru import logger

from bartleby.lib.consts import DEFAULT_MAX_WORKERS


def _add_project_arg(parser: argparse.ArgumentParser):
    """Add the standard --project argument."""
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name (defaults to active project)"
    )


def _add_verbose_arg(parser: argparse.ArgumentParser):
    """Add the standard --verbose argument."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show DEBUG messages)"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="bartleby",
        description="Bartleby, the Scrivener - A PDF processor that might refuse."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ready command
    ready_parser = subparsers.add_parser("ready", help="Configure Bartleby settings")

    # Project command
    project_parser = subparsers.add_parser("project", help="Manage projects")
    project_subparsers = project_parser.add_subparsers(dest="project_command", help="Project commands")

    project_create = project_subparsers.add_parser("create", help="Create a new project")
    project_create.add_argument("name", type=str, help="Project name")

    project_list = project_subparsers.add_parser("list", help="List all projects")

    project_use = project_subparsers.add_parser("use", help="Switch to an existing project")
    project_use.add_argument("name", type=str, help="Project name")

    project_info = project_subparsers.add_parser("info", help="Show project details")
    project_info.add_argument("name", type=str, nargs="?", default=None, help="Project name (defaults to active)")

    project_delete = project_subparsers.add_parser("delete", help="Delete a project")
    project_delete.add_argument("name", type=str, help="Project name")
    project_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    # Read command
    read_parser = subparsers.add_parser("read", help="Process PDF and HTML documents")
    read_parser.add_argument(
        "--files",
        "--pdfs",  # Backward compatibility
        dest="files",
        required=True,
        type=str,
        help="Path to a file or directory containing PDF/HTML files"
    )
    _add_project_arg(read_parser)
    read_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"Maximum number of worker threads (default: from config or {DEFAULT_MAX_WORKERS})"
    )
    read_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (e.g., claude-3-5-sonnet-20241022, default: from config)"
    )
    read_parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai"],
        default=None,
        help="LLM provider (anthropic or openai, default: from config)"
    )
    read_parser.add_argument(
        "--docling",
        action="store_true",
        help="Use Docling for layout-aware document conversion and structure-aware chunking"
    )
    _add_verbose_arg(read_parser)

    # Write command
    write_parser = subparsers.add_parser("write", help="Research agent for document investigation")
    _add_project_arg(write_parser)
    _add_verbose_arg(write_parser)

    # Book command
    book_parser = subparsers.add_parser("book", help="View research activity and findings")
    _add_project_arg(book_parser)
    book_subparsers = book_parser.add_subparsers(dest="book_command", help="Book commands")

    book_sessions = book_subparsers.add_parser("sessions", help="List research sessions")

    book_notes = book_subparsers.add_parser("notes", help="View research notes and findings")
    book_notes.add_argument(
        "--full",
        action="store_true",
        help="Show full note content instead of titles only"
    )
    book_notes.add_argument(
        "session",
        type=str,
        nargs="?",
        default=None,
        help="Filter by session name or uuid"
    )

    book_logs = book_subparsers.add_parser("logs", help="View tool call logs and token usage")
    book_logs.add_argument(
        "--session",
        type=str,
        default=None,
        help="Filter by session name or uuid"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ready":
        from bartleby.lib.console import send
        from bartleby.ready.main import main as ready_main

        send(message_type="SPLASH")
        ready_main()

    elif args.command == "project":
        _handle_project(args, project_parser)

    elif args.command == "read":
        from bartleby.lib.console import send
        from bartleby.project import get_project_db_path, resolve_active_db_path
        from bartleby.read.main import main as read_main

        send(message_type="SPLASH")

        if args.project:
            db_path = get_project_db_path(args.project)
        else:
            db_path = resolve_active_db_path()

        db_dir = db_path.parent

        # Create database if it doesn't exist
        if not db_path.exists():
            send(f"Creating database at {db_path}", "BIG")
            from bartleby.read.sqlite import create_db
            create_db(db_dir)

        read_main(
            db_path=db_path,
            pdf_path=args.files,
            max_workers=args.max_workers,
            model=args.model,
            provider=args.provider,
            verbose=args.verbose,
            use_docling=args.docling,
        )

    elif args.command == "write":
        from bartleby.project import get_project_db_path, resolve_active_db_path
        from bartleby.write.main import main as write_main

        if args.project:
            db_path = get_project_db_path(args.project)
        else:
            db_path = resolve_active_db_path()

        write_main(db_path=db_path, verbose=args.verbose)

    elif args.command == "book":
        from bartleby.project import get_project_db_path, resolve_active_db_path
        from bartleby.book.main import main as book_main

        if args.project:
            db_path = get_project_db_path(args.project)
        else:
            db_path = resolve_active_db_path()

        # Determine subcommand and options
        subcommand = args.book_command
        full = getattr(args, "full", False)
        session_filter = getattr(args, "session", None)

        book_main(
            db_path=db_path,
            subcommand=subcommand,
            full=full,
            session_filter=session_filter,
        )


def _handle_project_create(args, console):
    """Handle: bartleby project create <name>"""
    from bartleby.project import create_project

    try:
        project_dir = create_project(args.name)
        console.print(f"[bold green]Created project '{args.name}'[/bold green]")
        console.print(f"Location: [cyan]{project_dir}[/cyan]")
        console.print(f"Active project set to: [bold]{args.name}[/bold]")
    except (ValueError, FileExistsError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _handle_project_list(args, console):
    """Handle: bartleby project list"""
    from rich.table import Table

    from bartleby.project import list_projects

    projects = list_projects()
    if not projects:
        console.print("No projects found. Create one with `bartleby project create <name>`")
        return

    table = Table(title="Projects")
    table.add_column("", width=2)
    table.add_column("Name", style="bold")
    table.add_column("Database")

    for p in projects:
        marker = "*" if p["is_active"] else ""
        db_status = "[green]ready[/green]" if p["has_db"] else "[red]no db[/red]"
        table.add_row(marker, p["name"], db_status)

    console.print(table)


def _handle_project_use(args, console):
    """Handle: bartleby project use <name>"""
    from bartleby.project import set_active_project

    try:
        set_active_project(args.name)
        console.print(f"Active project set to: [bold]{args.name}[/bold]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _handle_project_info(args, console):
    """Handle: bartleby project info [name]"""
    from rich.table import Table

    from bartleby.project import get_active_project, get_project_info

    name = args.name or get_active_project()
    if not name:
        console.print("[red]No active project. Specify a name: `bartleby project info <name>`[/red]")
        sys.exit(1)

    try:
        info = get_project_info(name)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    table = Table(title=f"Project: {info['name']}")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    active_str = "yes" if info["is_active"] else "no"
    table.add_row("Active", active_str)
    table.add_row("Path", str(info["path"]))
    table.add_row("Database", "ready" if info["has_db"] else "missing")
    table.add_row("DB size", f"{info['db_size_mb']} MB")
    table.add_row("Documents", str(info["document_count"]))
    table.add_row("Report", "yes" if info["has_report"] else "no")
    table.add_row("Findings", str(info["findings_count"]))

    console.print(table)


def _handle_project_delete(args, console):
    """Handle: bartleby project delete <name>"""
    from rich.prompt import Confirm

    from bartleby.project import delete_project, get_active_project

    if not args.yes:
        confirmed = Confirm.ask(f"Delete project '{args.name}' and all its data?", default=False)
        if not confirmed:
            console.print("Cancelled.")
            return

    try:
        delete_project(args.name)
        console.print(f"[bold red]Deleted project '{args.name}'[/bold red]")
        if get_active_project() is None:
            console.print("[yellow]No active project. Set one with `bartleby project use <name>`[/yellow]")
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _handle_project(args, project_parser):
    """Route project subcommands to handlers."""
    from rich.console import Console

    console = Console()

    if not args.project_command:
        project_parser.print_help()
        sys.exit(1)

    handlers = {
        "create": _handle_project_create,
        "list": _handle_project_list,
        "use": _handle_project_use,
        "info": _handle_project_info,
        "delete": _handle_project_delete,
    }

    handler = handlers.get(args.project_command)
    if handler:
        handler(args, console)


if __name__ == "__main__":
    main()
