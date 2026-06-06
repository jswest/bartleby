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
from bartleby.config import PROJECTS_DIR
from bartleby.db.schema import DDL, EMBEDDING_DIM, SCHEMA_VERSION
from bartleby.lib.consts import EMBEDDING_MODEL


def project_db_path(project_name: str) -> Path:
    return PROJECTS_DIR / project_name / "bartleby.db"


def resolve_project_name(project_name: str | None) -> str:
    """Return ``project_name`` or the active project; raise if neither exists.

    The single resolve-or-fail path for an optional ``--project`` argument,
    shared by ``open_db`` and the CLI commands (``scribe``/``session``/
    ``logs``).
    """
    if project_name:
        return project_name
    from bartleby.project import get_active_project

    active = get_active_project()
    if not active:
        raise RuntimeError(
            "No active project. Run `bartleby project create <name>` to create one."
        )
    return active


def _attach(conn: apsw.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("PRAGMA busy_timeout = 5000")


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

    row = conn.cursor().execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
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
    finally:
        conn.close()
