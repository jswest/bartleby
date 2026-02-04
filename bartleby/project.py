"""Project management for Bartleby."""

import re
import shutil
from pathlib import Path

from bartleby.lib.config import PROJECTS_DIR, load_config, save_config_field
from bartleby.read.sqlite import create_db, get_connection

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


def get_project_db_path(name: str) -> Path:
    return PROJECTS_DIR / name / "bartleby.db"


def get_active_project() -> str | None:
    return load_config().get("active_project")


def set_active_project(name: str):
    project_dir = get_project_dir(name)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project '{name}' not found. Run `bartleby project list` to see available projects.")
    save_config_field("active_project", name)


def create_project(name: str) -> Path:
    validate_project_name(name)
    project_dir = get_project_dir(name)
    if project_dir.exists():
        raise FileExistsError(f"Project '{name}' already exists. Use `bartleby project use {name}` to switch to it.")

    project_dir.mkdir(parents=True, exist_ok=True)

    # Create the database
    create_db(project_dir)

    # Create archive directory
    (project_dir / "archive").mkdir(exist_ok=True)

    # Create book directory for write artifacts
    (project_dir / "book").mkdir(exist_ok=True)

    # Set as active project
    save_config_field("active_project", name)

    return project_dir


def delete_project(name: str):
    validate_project_name(name)
    project_dir = get_project_dir(name)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project '{name}' not found.")

    shutil.rmtree(project_dir)

    # Clear active project if it was this one
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

    db_path = project_dir / "bartleby.db"
    book_dir = project_dir / "book"
    report_path = book_dir / "report.md"
    findings_dir = book_dir / "findings"

    info = {
        "name": name,
        "path": project_dir,
        "is_active": get_active_project() == name,
        "has_db": db_path.exists(),
        "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 2) if db_path.exists() else 0,
        "document_count": 0,
        "has_report": report_path.exists(),
        "findings_count": len(list(findings_dir.glob("*.md"))) if findings_dir.exists() else 0,
    }

    if db_path.exists():
        try:
            conn = get_connection(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            info["document_count"] = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass

    return info


def resolve_active_db_path() -> Path:
    """Get the db_path for the active project, or raise a clear error."""
    active = get_active_project()
    if not active:
        raise RuntimeError("No active project. Run `bartleby project create <name>` to create one.")

    project_dir = get_project_dir(active)
    if not project_dir.exists():
        raise RuntimeError(
            f"Active project '{active}' not found on disk. "
            "Run `bartleby project list` to see available projects."
        )

    return get_project_db_path(active)
