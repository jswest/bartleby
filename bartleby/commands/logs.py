"""`bartleby logs` — pretty-print audit_logs rows for a session."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from bartleby.db.connection import open_db, resolve_project_name
from bartleby.lib import console
from bartleby.session import read_active_session_id


_ARGS_TRUNCATE = 80
_console = Console()


def _truncate(text: str | None, n: int = _ARGS_TRUNCATE) -> str:
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _format_duration(ms: int | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.2f}s"


def _resolve_session(conn, session: str | None, project_name: str):
    cur = conn.cursor()
    if session:
        row = cur.execute(
            "SELECT session_id, name FROM sessions WHERE name = ?", (session,)
        ).fetchone()
        if row is None:
            console.error(f"No session named '{session}'.")
            sys.exit(1)
        return row
    active_id = read_active_session_id(project_name)
    if active_id is not None:
        row = cur.execute(
            "SELECT session_id, name FROM sessions WHERE session_id = ?", (active_id,)
        ).fetchone()
        if row is not None:
            return row
    return cur.execute(
        "SELECT session_id, name FROM sessions ORDER BY session_id DESC LIMIT 1"
    ).fetchone()


def main(*, session: str | None = None, limit: int = 50, project: str | None = None) -> None:
    # SQLite reads a negative LIMIT as "unbounded", so `logs --limit -5` would
    # silently dump every row; 0 is meaningless. Reject anything below 1.
    if limit < 1:
        console.error("--limit must be a positive integer.")
        sys.exit(1)
    try:
        project_name = resolve_project_name(project)
    except (ValueError, RuntimeError) as e:
        console.error(str(e))
        sys.exit(1)

    conn = open_db(project_name)
    try:
        resolved = _resolve_session(conn, session, project_name)
        if resolved is None:
            _console.print("No sessions yet.")
            return
        session_id, session_name = resolved

        rows = list(conn.cursor().execute(
            "SELECT created_at, tool_name, args_json, duration_ms "
            "FROM audit_logs WHERE session_id = ? "
            "ORDER BY audit_log_id DESC LIMIT ?",
            (session_id, limit),
        ))
    finally:
        conn.close()

    rows.reverse()  # display in chronological order

    table = Table(
        title=f"audit_logs — session '{session_name}' (id={session_id})",
        show_lines=False,
    )
    table.add_column("time", style="dim", no_wrap=True)
    table.add_column("tool", style="bold")
    table.add_column("args")
    table.add_column("duration", justify="right", style="dim")

    if not rows:
        _console.print(f"No logged calls for session '{session_name}'.")
        return

    for created_at, tool_name, args_json, duration_ms in rows:
        table.add_row(
            created_at,
            tool_name,
            _truncate(args_json),
            _format_duration(duration_ms),
        )

    _console.print(table)
