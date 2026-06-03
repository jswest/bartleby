"""Shared lifecycle for ``skill/scripts/`` executables.

Each script defines ``parse_args(argv)`` and ``work(*, conn, args, session_id)``
and calls :func:`run`. The runner:

- Resolves the project (``--project`` or active).
- Opens the DB and validates the schema (via :func:`open_db`).
- Resolves or auto-creates the active session (skill scripts always run
  inside a session — see SPEC §5.4 / §6 intro).
- Times the call.
- Writes one ``audit_logs`` row per invocation (success or failure).
- Prints either the worker's dict result, or
  ``{"error": ..., "code": ...}`` on failure, to stdout.
- Exits 0 on success and 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable

from bartleby.db.audit import log_call
from bartleby.db.connection import open_db
from bartleby.project import get_active_project
from bartleby.session import ensure_active_session


class SkillError(Exception):
    """Raised by a skill ``work`` callback to surface a structured error.

    Anything in ``extra`` is merged into the stdout error envelope.
    """

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, separators=(",", ":"), default=str)
    sys.stdout.write("\n")


def run(
    *,
    tool_name: str,
    parse_args: Callable[[list[str] | None], argparse.Namespace],
    work: Callable[..., dict],
    argv: list[str] | None = None,
) -> None:
    # Skill scripts emit JSON to stdout; any third-party noise corrupts it.
    from bartleby.lib.quiet import setup_quiet_third_party
    setup_quiet_third_party(verbose=False)

    # Best-effort: guarantee the skill's scratch dir exists (mode 700) before the
    # agent writes a finding body there. Every realistic flow runs a skill command
    # before save_finding, so this races ahead of the agent's --body-file write.
    # Never let scratch setup break the actual tool call.
    try:
        from bartleby import config
        config.ensure_scratch_dir()
    except Exception:
        pass

    start = time.perf_counter()
    args = parse_args(argv)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    conn = None
    session_id: int | None = None
    result: dict | None = None
    error_envelope: dict | None = None

    try:
        project = args.project or get_active_project()
        if not project:
            raise SkillError(
                "NO_ACTIVE_PROJECT",
                "No active project. Run `bartleby project create <name>`.",
            )

        conn = open_db(project)
        session_id = ensure_active_session(project)
        result = work(conn=conn, args=args, session_id=session_id)
    except SkillError as e:
        error_envelope = {"error": e.message, "code": e.code, **e.extra}
    except Exception as e:  # noqa: BLE001 — catch-all by design
        error_envelope = {
            "error": f"{type(e).__name__}: {e}",
            "code": "INTERNAL_ERROR",
        }

    duration_ms = int((time.perf_counter() - start) * 1000)
    summary = (
        f"error: {error_envelope['code']}" if error_envelope else None
    )

    if conn is not None and session_id is not None:
        try:
            log_call(
                conn,
                session_id=session_id,
                tool_name=tool_name,
                args=args_dict,
                result_summary=summary,
                duration_ms=duration_ms,
            )
        except Exception:  # never let logging mask the real outcome
            pass
        conn.close()

    if error_envelope is not None:
        _print_json(error_envelope)
        sys.exit(1)

    _print_json(result)
