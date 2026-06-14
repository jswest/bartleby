"""Open and initialize Bartleby SQLite databases.

This module is the only place that opens a project DB and the only place that
runs the schema DDL. Everything else imports ``open_db`` (or ``init_db``) from
here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import apsw
import sqlite_vec

from bartleby import __version__ as BARTLEBY_VERSION
from bartleby.config import projects_dir
from bartleby.db.schema import DDL, EMBEDDING_DIM, SCHEMA_VERSION
from bartleby.lib.consts import EMBEDDING_MODEL


def project_db_path(project_name: str) -> Path:
    return projects_dir() / project_name / "bartleby.db"


def resolve_project_name(project_name: str | None) -> str:
    """Return ``project_name`` or the active project; raise if neither exists.

    The single resolve-or-fail path for an optional ``--project`` argument,
    shared by ``open_db`` and the CLI commands (``scribe``/``session``/
    ``logs``). Validates the name here, at the chokepoint, so a traversal
    string is rejected before any path is built from it (``open_db``, the
    skill runner, scribe archives, session pointers, logs).
    """
    from bartleby.project import get_active_project, validate_project_name

    if project_name:
        validate_project_name(project_name)
        return project_name

    active = get_active_project()
    if not active:
        raise RuntimeError(
            "No active project. Run `bartleby project create <name>` to create one."
        )
    return active


def _attach(conn: apsw.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    # busy_timeout MUST be set before `journal_mode = WAL`: that pragma itself
    # takes a database lock, so when several connections open the same project
    # DB concurrently (e.g. a handful of agents writing findings at once),
    # whichever runs the WAL pragma second would get an instant BusyError with
    # no timeout configured yet. Setting the timeout first makes every later
    # statement — the WAL pragma and the `BEGIN IMMEDIATE` at the mutating seam
    # — wait out a held lock instead of failing on first contact (#562).
    cur.execute("PRAGMA busy_timeout = 5000")
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")


def open_db(project_name: str | None = None) -> apsw.Connection:
    """Open the DB for a project (or the active project) and verify the schema.

    Loads sqlite-vec, enables FK checks, and raises if the schema version
    does not match ``SCHEMA_VERSION``.
    """
    name = resolve_project_name(project_name)
    db_path = project_db_path(name)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database does not exist at {db_path}. "
            f"Run `bartleby project create {name}` to create it."
        )

    conn = apsw.Connection(str(db_path))
    _attach(conn)

    try:
        row = conn.cursor().execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    except apsw.SQLError:
        # No `meta` table at all (a truncated/partial DB from an interrupted
        # create, or a foreign SQLite file). Give it the same friendly
        # "recreate the project" guidance as the missing-row case below rather
        # than leaking a raw "no such table" SQLError.
        row = None
    if row is None:
        raise RuntimeError(
            f"Database at {db_path} has no schema_version. Recreate the project."
        )
    db_version = int(row[0])
    if db_version != SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch for project '{name}': "
            f"database has version {db_version}, code expects {SCHEMA_VERSION}. "
            f"Run `bartleby project upgrade {name}` to upgrade in place; if that "
            "reports a non-additive bump, re-ingest the project instead."
        )

    return conn


def init_db(project_name: str) -> None:
    """Create a fresh database for ``project_name`` and populate ``meta``.

    Called by ``bartleby project create``. Raises ``FileExistsError`` if the
    database already exists; this is destructive-on-create, not idempotent.
    """
    db_path = project_db_path(project_name)
    if db_path.exists():
        raise FileExistsError(f"Database already exists at {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = apsw.Connection(str(db_path))
    try:
        _attach(conn)
        cur = conn.cursor()
        # apsw consumes multi-statement DDL by iterating the cursor.
        for _ in cur.execute(DDL):
            pass

        vec_version = cur.execute("SELECT vec_version()").fetchone()[0]
        now = datetime.now(timezone.utc).isoformat()
        meta_rows = [
            ("schema_version", str(SCHEMA_VERSION)),
            ("embedding_model", EMBEDDING_MODEL),
            ("embedding_dim", str(EMBEDDING_DIM)),
            ("sqlite_vec_version", vec_version),
            ("bartleby_version", BARTLEBY_VERSION),
            ("created_at", now),
        ]
        cur.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            meta_rows,
        )
    except BaseException:
        # A half-built DB is worse than none: the next `create` would hit the
        # FileExistsError guard above and refuse to retry, leaving the user
        # stuck. Unlink it so a retried create starts clean. The connection
        # must close first, or the file (and its WAL sidecars) stays locked.
        conn.close()
        db_path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()
