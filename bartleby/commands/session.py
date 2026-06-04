"""`bartleby session` — start, current, end, set."""

from __future__ import annotations

import sys

from rich.console import Console

from bartleby.db.connection import resolve_project_name
from bartleby.session import (
    SessionInfo,
    end_active_session,
    get_current_session,
    set_session_provenance,
    start_session,
)


_console = Console()


def _resolve_project(name: str | None) -> str:
    try:
        return resolve_project_name(name)
    except RuntimeError as e:
        _console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _print_session(info: SessionInfo, *, prefix: str = "") -> None:
    mem = "on" if info["memory_enabled"] else "off"
    backend = f"{info['harness'] or 'unknown'}/{info['model'] or 'unknown'}"
    line = (
        f"{prefix}[bold]{info['name']}[/bold] "
        f"(id={info['session_id']}, memory={mem}, backend={backend}, "
        f"started {info['created_at']})"
    )
    if info["ended_at"]:
        line += f" — ended {info['ended_at']}"
    _console.print(line)


def start(
    *, project: str | None, no_memory: bool,
    harness: str | None = None, model: str | None = None,
) -> None:
    project_name = _resolve_project(project)
    info = start_session(
        project_name, memory_enabled=not no_memory,
        harness=harness, model=model,
    )
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


def set_provenance(
    *, project: str | None, harness: str | None, model: str | None
) -> None:
    project_name = _resolve_project(project)
    if harness is None and model is None:
        _console.print("[red]Pass --model and/or --harness to set.[/red]")
        sys.exit(1)
    info = set_session_provenance(project_name, model=model, harness=harness)
    if info is None:
        _console.print("No active session to update.")
        return
    _print_session(info, prefix="Updated session ")
