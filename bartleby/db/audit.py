"""Append-only audit log for skill-script invocations.

Skill scripts call ``log_call`` once at the end of their work. Prose
progress goes to stderr; the persistent record lands here. The agent never
reads from ``audit_logs`` — only the user, via ``bartleby logs``.
"""

from __future__ import annotations

import json
from typing import Any

import apsw


def log_call(
    conn: apsw.Connection,
    *,
    session_id: int | None,
    tool_name: str,
    args: dict[str, Any] | None = None,
    result_summary: str | None = None,
    duration_ms: int | None = None,
) -> int:
    """Insert one row into ``audit_logs`` and return its ``audit_log_id``.

    ``args`` is JSON-encoded before storage. ``session_id`` may be ``None``
    if no session is active (the FK is ``ON DELETE SET NULL``).
    """
    args_json = json.dumps(args, default=str) if args is not None else None
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audit_logs "
        "(session_id, tool_name, args_json, result_summary, duration_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, tool_name, args_json, result_summary, duration_ms),
    )
    return conn.last_insert_rowid()
