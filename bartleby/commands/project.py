"""`bartleby project` — create / list / use / info / delete."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from bartleby.project import (
    create_project,
    delete_project,
    get_active_project,
    get_project_info,
    list_projects,
    set_active_project,
)


_console = Console()


def create(*, name: str) -> None:
    try:
        project_dir = create_project(name)
    except (ValueError, FileExistsError) as e:
        _console.print(f"[red]{e}[/red]")
        sys.exit(1)
    _console.print(f"[bold green]Created project '{name}'[/bold green]")
    _console.print(f"Location: [cyan]{project_dir}[/cyan]")
    _console.print(f"Active project set to: [bold]{name}[/bold]")


def list_(*, _: None = None) -> None:
    projects = list_projects()
    if not projects:
        _console.print(
            "No projects found. Create one with `bartleby project create <name>`"
        )
        return

    table = Table(title="Projects")
    table.add_column("", width=2)
    table.add_column("Name", style="bold")
    table.add_column("Database")
    for p in projects:
        marker = "*" if p["is_active"] else ""
        db_status = "[green]ready[/green]" if p["has_db"] else "[red]no db[/red]"
        table.add_row(marker, p["name"], db_status)
    _console.print(table)


def use(*, name: str) -> None:
    try:
        set_active_project(name)
    except FileNotFoundError as e:
        _console.print(f"[red]{e}[/red]")
        sys.exit(1)
    _console.print(f"Active project set to: [bold]{name}[/bold]")


def info(*, name: str | None) -> None:
    name = name or get_active_project()
    if not name:
        _console.print(
            "[red]No active project. Specify a name: `bartleby project info <name>`[/red]"
        )
        sys.exit(1)
    try:
        i = get_project_info(name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        _console.print(f"[red]{e}[/red]")
        sys.exit(1)

    table = Table(title=f"Project: {i['name']}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Active", "yes" if i["is_active"] else "no")
    table.add_row("Path", str(i["path"]))
    table.add_row("Database", "ready" if i["has_db"] else "missing")
    if i["has_db"]:
        table.add_row("DB size", f"{i['db_size_mb']} MB")
        table.add_row("Schema version", str(i["schema_version"]))
        table.add_row("Embedding model", str(i["embedding_model"]))
        table.add_row("Documents", str(i["document_count"]))
        table.add_row("Sessions", str(i["session_count"]))
        table.add_row("Findings", str(i["finding_count"]))
        c = i["chunk_counts"]
        table.add_row(
            "Chunks",
            f"{c['document']} document  /  {c['summary']} summary  /  "
            f"{c['finding']} finding",
        )
    _console.print(table)


def delete(*, name: str, yes: bool) -> None:
    if not yes:
        if not Confirm.ask(
            f"Delete project '{name}' and all its data?", default=False
        ):
            _console.print("Cancelled.")
            return
    try:
        delete_project(name)
    except (ValueError, FileNotFoundError) as e:
        _console.print(f"[red]{e}[/red]")
        sys.exit(1)
    _console.print(f"[bold red]Deleted project '{name}'[/bold red]")
    if get_active_project() is None:
        _console.print(
            "[yellow]No active project. Set one with `bartleby project use <name>`[/yellow]"
        )
