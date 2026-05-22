"""Session state for Bartleby.

Sessions are first-class rows in ``sessions`` plus a small ``.active_session``
file in the project directory that points at the active one. The skill never
opens the DB just to figure out which session is current — it reads the file.

If the file is missing and a script needs a session, ``ensure_active_session``
silently creates one with default settings. Sessions are bookkeeping, not a
gate (SPEC §5.4).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import apsw

from bartleby.db.connection import open_db
from bartleby.project import get_project_dir


_ADJECTIVES = (
    "amber", "bold", "brave", "bright", "calm", "clever", "dappled", "deep",
    "eager", "fair", "fierce", "gentle", "golden", "happy", "humble", "jolly",
    "keen", "kind", "lively", "lucky", "merry", "mighty", "noble", "proud",
    "quick", "quiet", "rapid", "rugged", "sage", "sharp", "silent", "silver",
    "sleek", "soft", "stark", "steady", "stout", "strong", "sturdy", "sunny",
    "swift", "tender", "tidy", "tough", "true", "vast", "warm", "wild", "wise",
    "witty",
)

_NOUNS = (
    "ash", "badger", "beacon", "birch", "bridge", "brook", "cedar", "cliff",
    "cloud", "coral", "creek", "dawn", "delta", "dune", "ember", "fern",
    "field", "fjord", "forge", "glade", "glen", "grove", "harbor", "haven",
    "hearth", "heron", "hollow", "holly", "ivy", "knoll", "lantern", "ledge",
    "marsh", "meadow", "mesa", "moor", "oak", "orchard", "peak", "pine",
    "prairie", "reef", "ridge", "river", "shore", "slope", "stone", "summit",
    "thicket", "tide", "vale", "valley", "willow",
)


class SessionInfo(TypedDict):
    session_id: int
    name: str
    memory_enabled: bool
    created_at: str
    ended_at: str | None


def generate_name() -> str:
    return f"{secrets.choice(_ADJECTIVES)}-{secrets.choice(_NOUNS)}"


def _active_session_file(project_name: str) -> Path:
    return get_project_dir(project_name) / ".active_session"


def read_active_session_id(project_name: str) -> int | None:
    f = _active_session_file(project_name)
    if not f.exists():
        return None
    try:
        return int(f.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def write_active_session_id(project_name: str, session_id: int) -> None:
    f = _active_session_file(project_name)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(str(session_id), encoding="utf-8")


def clear_active_session(project_name: str) -> None:
    f = _active_session_file(project_name)
    if f.exists():
        f.unlink()


def _row_to_info(row) -> SessionInfo:
    return SessionInfo(
        session_id=row[0],
        name=row[1],
        memory_enabled=bool(row[2]),
        created_at=row[3],
        ended_at=row[4],
    )


def _fetch_session(conn, session_id: int) -> SessionInfo | None:
    row = conn.cursor().execute(
        "SELECT session_id, name, memory_enabled, created_at, ended_at "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return _row_to_info(row) if row else None


def start_session(
    project_name: str,
    *,
    memory_enabled: bool = True,
    max_attempts: int = 32,
) -> SessionInfo:
    """Insert a new session, mark it active, return its info.

    Generates a memorable ``adjective-noun`` name; on collision (UNIQUE),
    retries with a fresh draw up to ``max_attempts`` times.
    """
    conn = open_db(project_name)
    try:
        cur = conn.cursor()
        for _ in range(max_attempts):
            name = generate_name()
            try:
                cur.execute(
                    "INSERT INTO sessions (name, memory_enabled) VALUES (?, ?)",
                    (name, 1 if memory_enabled else 0),
                )
            except apsw.ConstraintError:
                continue
            session_id = conn.last_insert_rowid()
            write_active_session_id(project_name, session_id)
            return _fetch_session(conn, session_id)
        raise RuntimeError(
            f"Could not generate a unique session name after {max_attempts} tries; "
            f"the namespace may be exhausted."
        )
    finally:
        conn.close()


def get_current_session(project_name: str) -> SessionInfo | None:
    """Return the active session for ``project_name`` if one is set, else None.

    The on-disk pointer may be stale (e.g. the session was deleted out from
    under us). In that case we silently treat it as missing.
    """
    sid = read_active_session_id(project_name)
    if sid is None:
        return None
    conn = open_db(project_name)
    try:
        info = _fetch_session(conn, sid)
    finally:
        conn.close()
    if info is None:
        clear_active_session(project_name)
    return info


def end_active_session(project_name: str) -> SessionInfo | None:
    """Mark the active session ended (cosmetic) and clear the pointer.

    Returns the ended session info, or ``None`` if no session was active.
    """
    sid = read_active_session_id(project_name)
    if sid is None:
        return None
    conn = open_db(project_name)
    try:
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
            (now, sid),
        )
        info = _fetch_session(conn, sid)
    finally:
        conn.close()
    clear_active_session(project_name)
    return info


def ensure_active_session(project_name: str) -> int:
    """Return the active session_id, creating one with defaults if missing.

    For use by skill scripts that must always have a session to attribute
    findings and audit-log entries to.
    """
    sid = read_active_session_id(project_name)
    if sid is not None:
        # Validate the pointer is still live.
        conn = open_db(project_name)
        try:
            row = conn.cursor().execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
        finally:
            conn.close()
        if row:
            return sid
        clear_active_session(project_name)

    info = start_session(project_name, memory_enabled=True)
    return info["session_id"]
