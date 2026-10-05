"""Project management for Bartleby.

Projects are directories under ~/.bartleby/projects/ holding one SQLite DB
plus an ``archive/`` of ingested originals. The active project is tracked in
the user config.
"""

import os
import re
import shutil
from collections import Counter
from pathlib import Path

import apsw

from bartleby.config import load_config, projects_dir, save_config_field
from bartleby.db.connection import _attach, init_db, open_db, project_db_path
from bartleby.db.schema import ALLOWED_SOURCE_KINDS

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def validate_project_name(name: str):
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name: '{name}'. "
            "Must start with a letter or digit, contain only letters, digits, hyphens, "
            "and underscores, and be 1-64 characters long."
        )


def get_project_dir(name: str) -> Path:
    return projects_dir() / name


def get_active_project() -> str | None:
    return load_config().get("active_project")


def set_active_project(name: str):
    validate_project_name(name)
    project_dir = get_project_dir(name)
    if not project_dir.exists():
        raise FileNotFoundError(
            f"Project '{name}' not found. "
            "Run `bartleby project list` to see available projects."
        )
    save_config_field("active_project", name)


def create_project(name: str) -> Path:
    validate_project_name(name)
    project_dir = get_project_dir(name)
    if project_dir.exists():
        raise FileExistsError(
            f"Project '{name}' already exists. "
            f"Use `bartleby project use {name}` to switch to it."
        )

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "archive").mkdir(exist_ok=True)
    init_db(name)
    save_config_field("active_project", name)
    return project_dir


def delete_project(name: str):
    validate_project_name(name)
    project_dir = get_project_dir(name)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project '{name}' not found.")

    shutil.rmtree(project_dir)

    if get_active_project() == name:
        save_config_field("active_project", None)


def rename_project(old: str, new: str) -> bool:
    """Rename project ``old`` to ``new``: directory, archive paths, active pointer.

    ``documents.file_path`` / ``images.file_path`` hold absolute archive paths
    that embed the project name, so after moving the directory their prefix is
    rewritten in one transaction. If that rewrite fails, the directory is moved
    back and the error re-raised, leaving the project as it was. Returns
    whether the active-project pointer moved with it.
    """
    validate_project_name(old)
    validate_project_name(new)
    if old == new:
        raise ValueError(f"Project is already named '{old}'.")
    old_dir, new_dir = get_project_dir(old), get_project_dir(new)
    if not old_dir.exists():
        raise FileNotFoundError(f"Project '{old}' not found.")
    if new_dir.exists():
        raise FileExistsError(f"Project '{new}' already exists.")

    old_dir.rename(new_dir)
    try:
        db_path = project_db_path(new)
        if db_path.exists():
            _swap_archive_prefix(
                db_path,
                str(old_dir / "archive") + os.sep,
                str(new_dir / "archive") + os.sep,
            )
    except BaseException:
        new_dir.rename(old_dir)
        raise

    if get_active_project() != old:
        return False
    save_config_field("active_project", new)
    return True


def _swap_archive_prefix(db_path: Path, old_prefix: str, new_prefix: str) -> None:
    """Rewrite ``old_prefix`` → ``new_prefix`` on every archive ``file_path``.

    Raises (rolling the transaction back) if any row is left outside
    ``new_prefix`` — e.g. stored under a different spelling of
    ``BARTLEBY_HOME`` (symlink, env override) — so a rename never reports
    success while leaving archive paths dangling.
    """
    conn = apsw.Connection(str(db_path))
    try:
        _attach(conn)
        with conn:
            cur = conn.cursor()
            stray = 0
            for table in ("documents", "images"):
                cur.execute(
                    f"UPDATE {table} SET file_path = ? || substr(file_path, ?) "
                    "WHERE substr(file_path, 1, ?) = ?",
                    (new_prefix, len(old_prefix) + 1, len(old_prefix), old_prefix),
                )
                stray += cur.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    "WHERE substr(file_path, 1, ?) != ?",
                    (len(new_prefix), new_prefix),
                ).fetchone()[0]
            if stray:
                raise ValueError(
                    f"{stray} archive path(s) don't start with {old_prefix} "
                    "(stored under a different BARTLEBY_HOME spelling, or the "
                    "project was moved by hand); nothing was renamed."
                )
    finally:
        conn.close()


def list_projects() -> list[dict]:
    root = projects_dir()
    if not root.exists():
        return []

    active = get_active_project()
    projects = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            db_path = entry / "bartleby.db"
            projects.append({
                "name": entry.name,
                "path": entry,
                "has_db": db_path.exists(),
                "is_active": entry.name == active,
            })
    return projects


def get_project_info(name: str) -> dict:
    validate_project_name(name)
    project_dir = get_project_dir(name)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project '{name}' not found.")

    db_path = project_db_path(name)
    info = {
        "name": name,
        "path": project_dir,
        "is_active": get_active_project() == name,
        "has_db": db_path.exists(),
        "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 2)
        if db_path.exists() else 0,
        "schema_version": None,
        "embedding_model": None,
        "document_count": 0,
        "session_count": 0,
        "finding_count": 0,
        "chunk_counts": {kind: 0 for kind in ALLOWED_SOURCE_KINDS},
        "failed_ingests": {"total": 0, "capped": 0},
        "ingest_runs": {"count": 0, "first_started": None, "last_finished": None},
        "source_dirs": [],
    }

    if not db_path.exists():
        return info

    conn = open_db(name)
    try:
        cur = conn.cursor()
        meta = dict(cur.execute("SELECT key, value FROM meta"))
        info["schema_version"] = meta.get("schema_version")
        info["embedding_model"] = meta.get("embedding_model")

        info["document_count"] = cur.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        info["session_count"] = cur.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        info["finding_count"] = cur.execute(
            "SELECT COUNT(*) FROM findings"
        ).fetchone()[0]
        for kind, count in cur.execute(
            "SELECT source_kind, COUNT(*) FROM chunks GROUP BY source_kind"
        ):
            info["chunk_counts"][kind] = count

        # Per-unit ingest failures (parse/caption/summary) that never resolved.
        # Surfaced so a capped, permanently-skipped unit can't read as done.
        from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS
        info["failed_ingests"] = {
            "total": cur.execute(
                "SELECT COUNT(*) FROM failed_ingests"
            ).fetchone()[0],
            "capped": cur.execute(
                "SELECT COUNT(*) FROM failed_ingests WHERE attempts >= ?",
                (MAX_INGEST_ATTEMPTS,),
            ).fetchone()[0],
        }

        # Ingest provenance (#686). last_finished is the newest run's own
        # finished_at — NULL there means that run is in progress or was cut off.
        count, first_started = cur.execute(
            "SELECT COUNT(*), MIN(started_at) FROM ingests"
        ).fetchone()
        last = cur.execute(
            "SELECT finished_at FROM ingests ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        info["ingest_runs"] = {
            "count": count,
            "first_started": first_started,
            "last_finished": last[0] if last else None,
        }
        # Docs per source directory, top-level rows only (#254 sections share
        # their container's path). A NULL source_path groups under None.
        info["source_dirs"] = Counter(
            str(Path(sp).parent) if sp else None
            for (sp,) in cur.execute(
                "SELECT source_path FROM documents "
                "WHERE parent_document_id IS NULL"
            )
        ).most_common()
    finally:
        conn.close()

    return info


def get_document_sources(name: str) -> list[dict]:
    """One row per top-level document (#254 sections excluded) with where it
    came from (#686), sorted by source_path then file_name; NULL paths last."""
    validate_project_name(name)
    conn = open_db(name)
    try:
        rows = conn.cursor().execute(
            "SELECT document_id, file_name, source_path, created_at "
            "FROM documents WHERE parent_document_id IS NULL "
            "ORDER BY source_path IS NULL, source_path, file_name"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"document_id": d, "file_name": f, "source_path": s, "created_at": c}
        for d, f, s, c in rows
    ]
