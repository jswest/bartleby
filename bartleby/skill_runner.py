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

A script that mutates the DB passes ``mutates=True`` to :func:`run`; the runner
then wraps the ``work`` call in a single ``with conn:`` transaction so the whole
mutation commits or rolls back atomically. Read-only scripts (the default) open
no transaction. The audit write always lands, transaction or not — including
when a mutation rolled back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

from bartleby.db.audit import log_call
from bartleby.db.connection import open_db
from bartleby.project import get_active_project
from bartleby.session import ensure_active_session, ensure_named_session


class SkillError(Exception):
    """Raised by a skill ``work`` callback to surface a structured error.

    Anything in ``extra`` is merged into the stdout error envelope.
    """

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def build_arg_parser(
    prog: str, description: str | None = None
) -> argparse.ArgumentParser:
    """Build a skill script's ``ArgumentParser``.

    Pass the module docstring as ``description`` (i.e. ``__doc__``). It's
    rendered raw — not reflowed — so the ``Output:`` JSON block in the
    docstring survives intact, which means ``--help`` shows the agent both the
    arguments *and* the response shape. Agents read SKILL.md and ``--help``;
    the JSON contract otherwise lives only in docstrings they never see (issue
    #47). The raw formatter must travel with the docstring, so it lives here
    rather than at each of the ~18 call sites.
    """
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, separators=(",", ":"), default=str)
    sys.stdout.write("\n")


def run(
    *,
    tool_name: str,
    parse_args: Callable[[list[str] | None], argparse.Namespace],
    work: Callable[..., dict],
    argv: list[str] | None = None,
    mutates: bool = False,
) -> None:
    # Skill scripts emit JSON to stdout; any third-party noise corrupts it.
    # The only HF model the skill path loads is the embedding model (semantic
    # search); offline mode is gated on it being cached (issue #88).
    from bartleby.lib.consts import EMBEDDING_MODEL
    from bartleby.lib.quiet import setup_quiet_third_party
    setup_quiet_third_party(verbose=False, required_models=(EMBEDDING_MODEL,))

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

    conn = None
    session_id: int | None = None
    result: dict | None = None
    error_envelope: dict | None = None
    args_dict: dict[str, Any] = {}

    try:
        # parse_args lives inside the try so an argparse usage error
        # (a non-zero ``SystemExit`` from ``ArgumentParser.error()``) becomes
        # the JSON envelope every other failure already emits, rather than
        # argparse's raw stderr usage dump — agents parse one shape (issue
        # #402). ``--help`` exits 0 and is re-raised untouched below.
        args = parse_args(argv)
        args_dict = {k: v for k, v in vars(args).items() if v is not None}

        project = args.project or get_active_project()
        if not project:
            raise SkillError(
                "NO_ACTIVE_PROJECT",
                "No active project. Run `bartleby project create <name>`.",
            )

        conn = open_db(project)
        # A non-agent caller (the web UI) sets BARTLEBY_SESSION_NAME to pin its
        # own durable, memory-enabled session by name. This deliberately bypasses
        # the .active_session pointer so the web never hijacks — or is hijacked
        # by — whichever session an agent has active. Agents leave it unset and
        # get the normal active-session behavior.
        session_name = os.environ.get("BARTLEBY_SESSION_NAME")
        if session_name:
            session_id = ensure_named_session(project, session_name)
        else:
            session_id = ensure_active_session(project)
        if mutates:
            # One transaction at the seam: a mutating script's entire ``work``
            # commits or rolls back atomically. apsw's ``with conn:`` is a
            # deferred transaction (no write lock until the first write), and
            # the ``with conn:`` blocks inside the chunk helpers nest under it
            # as savepoints rather than committing independently — so a partial
            # write (e.g. chunks deleted but the row delete then raises) leaves
            # nothing behind. The audit ``log_call`` below stays OUTSIDE this
            # block on purpose: a rolled-back mutation must still record that it
            # was attempted and failed.
            with conn:
                result = work(conn=conn, args=args, session_id=session_id)
        else:
            result = work(conn=conn, args=args, session_id=session_id)
    except SystemExit as e:
        # argparse raises SystemExit: code 0 for ``--help``/``-h`` (let it
        # through unchanged), non-zero for a usage/argument error (turn it into
        # the envelope). parse_args fails before open_db, so there's no
        # conn/session to log — this falls through to the shared emit path below.
        if not e.code:  # None or 0 → clean exit (e.g. --help)
            raise
        error_envelope = {
            "error": "Invalid arguments. See --help for usage.",
            "code": "USAGE_ERROR",
        }
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
