"""Shared lifecycle for the ``bartleby/skill_scripts/`` executables.

Each script defines ``parse_args(argv)`` and ``work(*, conn, args, session_id)``
and calls :func:`run`. The runner:

- Resolves the project (``--project`` or active).
- Opens the DB and validates the schema (via :func:`open_db`).
- Resolves or auto-creates the active session (skill scripts always run
  inside a session — see SPEC §5.4 / §6 intro).
- Times the call.
- Writes one ``audit_logs`` row per invocation (success or failure) once a
  session is resolved. A failure *before* session resolution (no active
  project, DB open/schema error, or session resolution itself raising) has no
  session to attribute the call to, so it surfaces only in the error envelope;
  the DB connection is still closed on every path where it was opened.
- Prints either the worker's dict result, or
  ``{"error": ..., "code": ...}`` on failure, to stdout. On failure the error
  is *also* echoed as ``code: message`` to stderr (issue #421) so a stdout-only
  consumer can distinguish a tool error from an empty result; the stdout JSON
  envelope is unchanged.
- Exits 0 on success and 1 on failure.

A script that mutates the DB passes ``mutates=True`` to :func:`run`; the runner
then wraps the ``work`` call in a single ``BEGIN IMMEDIATE`` transaction so the
whole mutation commits or rolls back atomically. ``BEGIN IMMEDIATE`` (rather
than apsw's deferred ``with conn:``) takes the write lock up front so concurrent
writers serialize under ``busy_timeout`` instead of failing the read-to-write
upgrade with an instant ``BusyError`` (see the comment at the seam). Read-only
scripts (the default) open no transaction. The audit write always lands,
transaction or not — including when a mutation rolled back.
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
from bartleby.session import (
    ensure_active_session,
    ensure_named_session,
    ensure_session_by_run_key,
    run_echo,
)


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
    rather than at each of the ~two-dozen skill-script call sites.

    ``--project`` is declared here too: ``run()`` unconditionally reads
    ``args.project`` to resolve the corpus, so it is part of the runner's
    contract with every script (issue #432). Each script's parser inherits it
    from this factory rather than redeclaring the byte-identical line.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", type=str, default=None)
    # The per-conversation run id (#547). An agent mints it once via `skill
    # session new` and passes it on every later call so the work attaches to the
    # right run; like --project, run() reads it for every script, so it lives
    # here rather than in each parser.
    parser.add_argument("--run", type=str, default=None)
    return parser


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, separators=(",", ":"), default=str)
    sys.stdout.write("\n")


def _emit_error(envelope: dict) -> None:
    """Emit an error envelope on BOTH channels, then exit non-zero.

    The stdout JSON envelope is the machine contract and stays byte-for-byte
    unchanged. We *also* echo ``code: message`` to stderr (issue #421): today a
    stdout-only consumer can't tell a tool error from an empty result, and an
    error is prose — it belongs on stderr alongside the existing
    progress/prose-to-stderr split. This is the single error-emission point so
    the two channels never drift; individual scripts never write stderr
    themselves.
    """
    _print_json(envelope)
    sys.stderr.write(f"{envelope['code']}: {envelope['error']}\n")
    sys.exit(1)


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

    # Only the argparse call gets the USAGE_ERROR treatment: a non-zero
    # SystemExit from parse_args (argparse's usage error) becomes the JSON
    # envelope instead of a raw stderr dump, while a clean exit (code 0, e.g.
    # --help) re-raises untouched so help still exits 0. Narrowed to this one
    # call so a SystemExit raised anywhere else in the try (open_db, work,
    # teardown — or a library calling sys.exit) is NOT mislabeled USAGE_ERROR;
    # it surfaces as INTERNAL_ERROR via the catch-all below.
    # See docs/decisions/GH-0402-argparse-json-envelope-0001.md.
    try:
        args = parse_args(argv)
    except SystemExit as e:
        if not e.code:  # None or 0 → clean exit (e.g. --help); re-raise untouched
            raise
        _emit_error({
            "error": "Invalid arguments. See --help for usage.",
            "code": "USAGE_ERROR",
        })

    try:
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
        elif getattr(args, "run", None):
            # Agent carried a per-conversation run id (#547): resolve (or create)
            # the run bound to it. This is the precise path; the marker below is
            # only the fallback for a call that forgot to pass --run. getattr (not
            # args.run) because a parser that didn't come from build_arg_parser
            # has no `run` attribute — true of bare test parsers.
            session_id = ensure_session_by_run_key(project, args.run)
        else:
            session_id = ensure_active_session(project)
        if mutates:
            # One transaction at the seam: a mutating script's entire ``work``
            # commits or rolls back atomically. We open it with an explicit
            # ``BEGIN IMMEDIATE`` rather than apsw's ``with conn:`` (a *deferred*
            # transaction). A deferred txn takes no write lock until its first
            # write, so a ``work`` that reads before writing hits the SQLite
            # read-to-write upgrade trap: the upgrade returns ``SQLITE_BUSY``
            # immediately, *ignoring* ``busy_timeout`` (the timeout governs the
            # initial lock acquisition, not a mid-transaction upgrade). With a
            # handful of concurrent finding writes into one session that meant a
            # spurious first-attempt ``BusyError`` instead of serializing.
            # Taking the write lock up front makes ``busy_timeout`` (set in
            # ``connection._attach``) govern the wait, so concurrent writers
            # block and serialize. The ``with conn:`` blocks inside the chunk
            # helpers still nest under this open transaction as savepoints
            # rather than committing independently, so a partial write (e.g.
            # chunks deleted but the row delete then raises) leaves nothing
            # behind. The audit ``log_call`` below stays OUTSIDE this block on
            # purpose: a rolled-back mutation must still record that it was
            # attempted and failed.
            conn.cursor().execute("BEGIN IMMEDIATE")
            try:
                result = work(conn=conn, args=args, session_id=session_id)
            except BaseException:
                conn.cursor().execute("ROLLBACK")
                raise
            else:
                conn.cursor().execute("COMMIT")
        else:
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

    if conn is not None:
        # Every successful result echoes the current run (#547) under a "run"
        # key — id, name, model, and whether the model was self-reported — so
        # the agent can see and keep its run_key. Errors keep the bare error
        # envelope; the run row is committed by now (a mutating work's
        # `with conn:` has exited), so the read sees it.
        if error_envelope is None and isinstance(result, dict) and session_id is not None:
            result = {**result, "run": run_echo(conn, session_id)}
        # log_call needs a resolved session_id (its FK target); close must run
        # on every opened path regardless, or the conn leaks. See module docstring.
        if session_id is not None:
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
        _emit_error(error_envelope)

    _print_json(result)
