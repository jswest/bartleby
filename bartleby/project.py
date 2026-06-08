"""Project management for Bartleby.

Projects are directories under ~/.bartleby/projects/ holding one SQLite DB
plus an ``archive/`` of ingested originals. The active project is tracked in
the user config.
"""

import re
import shutil
from pathlib import Path

from bartleby.config import PROJECTS_DIR, load_config, save_config_field
from bartleby.db.connection import init_db, open_db, project_db_path
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
    return PROJECTS_DIR / name


def get_active_project() -> str | None:
    return load_config().get("active_project")


def set_active_project(name: str):
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


def list_projects() -> list[dict]:
    if not PROJECTS_DIR.exists():
        return []

    active = get_active_project()
    projects = []
    for entry in sorted(PROJECTS_DIR.iterdir()):
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
    finally:
        conn.close()

    return info
