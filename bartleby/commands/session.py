"""`bartleby session` — start, current, end."""

from __future__ import annotations

import sys

from rich.console import Console

from bartleby.project import get_active_project
from bartleby.session import (
    SessionInfo,
    end_active_session,
    get_current_session,
    start_session,
)


_console = Console()


def _resolve_project(name: str | None) -> str:
    project = name or get_active_project()
    if not project:
        _console.print(
            "[red]No active project. Run `bartleby project create <name>`.[/red]"
        )
        sys.exit(1)
    return project


def _print_session(info: SessionInfo, *, prefix: str = "") -> None:
    mem = "on" if info["memory_enabled"] else "off"
    line = (
        f"{prefix}[bold]{info['name']}[/bold] "
        f"(id={info['session_id']}, memory={mem}, started {info['created_at']})"
    )
    if info["ended_at"]:
        line += f" — ended {info['ended_at']}"
    _console.print(line)


def start(*, project: str | None, no_memory: bool) -> None:
    project_name = _resolve_project(project)
    info = start_session(project_name, memory_enabled=not no_memory)
    _print_session(info, prefix="Started session ")


def current(*, project: str | None) -> None:
    project_name = _resolve_project(project)
    info = get_current_session(project_name)
    if info is None:
        _console.print("No active session.")
        return
    _print_session(info, prefix="Active session: ")


def end(*, project: str | None) -> None:
    project_name = _resolve_project(project)
    info = end_active_session(project_name)
    if info is None:
        _console.print("No active session to end.")
        return
    _print_session(info, prefix="Ended session ")
