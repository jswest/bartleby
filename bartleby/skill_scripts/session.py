#!/usr/bin/env python3
"""session — manage the per-conversation research run (#547).

``bartleby skill session new`` is your **first** call in a conversation. It
mints a run id (a UUID), starts a fresh run, and returns the id. Pass that id as
``--run <uuid>`` on every later skill call so all your work attaches to this one
run. New conversation → new ``session new`` → new run; the two never tangle.

Optionally tell it which model you are with ``--model`` — recorded best-effort
as a self-reported claim and surfaced everywhere as "Set by LLM", so omit it if
you don't know your own name (common for local models). ``--no-memory`` starts
the run with memory off, for blind eval isolation.

Unlike the other scripts this one *defines* a run rather than operating inside
one, so it doesn't take ``--run`` and runs its own minimal flow (no active-
session resolution); it still emits the standard JSON envelope.

Output (subcommand ``new``):
    {
      "created": true,
      "run": {
        "run_key": str,          # pass this as --run <uuid> on later calls
        "session_id": int,
        "session_name": str,
        "model": str|null,
        "memory_enabled": bool,
        "model_set_by_llm": bool
      }
    }
"""

from __future__ import annotations

import argparse
import time
import uuid

from bartleby.db.audit import log_call
from bartleby.db.connection import open_db
from bartleby.project import get_active_project
from bartleby.session import run_echo, start_session

# The runner's envelope helpers are the JSON contract every skill speaks; reuse
# them so `session new` can't drift from it, even though it bypasses run().
from bartleby.skill_runner import _emit_error, _print_json


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="session",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand", required=True)
    new = sub.add_parser("new", help="Mint a run id and start a new run.")
    new.add_argument("--project", type=str, default=None)
    new.add_argument(
        "--model", type=str, default=None,
        help="Which model you are (self-reported; recorded as 'Set by LLM').",
    )
    new.add_argument(
        "--no-memory", action="store_true", dest="no_memory",
        help="Start the run with memory off (blind eval isolation).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        if not e.code:  # clean exit (e.g. --help) — re-raise untouched
            raise
        _emit_error({
            "error": "Invalid arguments. See --help for usage.",
            "code": "USAGE_ERROR",
        })

    start = time.perf_counter()
    project = args.project or get_active_project()
    if not project:
        _emit_error({
            "error": "No active project. Run `bartleby project create <name>`.",
            "code": "NO_ACTIVE_PROJECT",
        })

    # `new` is the only subcommand the parser accepts today.
    info = start_session(
        project,
        memory_enabled=not args.no_memory,
        model=args.model,
        run_key=str(uuid.uuid4()),
    )
    duration_ms = int((time.perf_counter() - start) * 1000)

    # start_session owns and closes its own connection by design, so reopen here
    # to echo the run and record the audit row. Deliberate — don't fold this back
    # into start_session.
    conn = open_db(project)
    try:
        echo = run_echo(conn, info["session_id"])
        try:
            log_call(
                conn,
                session_id=info["session_id"],
                tool_name="session_new",
                args={k: v for k, v in vars(args).items() if v is not None},
                duration_ms=duration_ms,
            )
        except Exception:  # never let logging mask a created run
            pass
    finally:
        conn.close()

    _print_json({"created": True, "run": echo})
