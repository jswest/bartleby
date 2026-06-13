"""Session state for Bartleby.

Sessions are first-class rows in ``sessions`` plus a small ``.active_session``
file in the project directory that points at the active one. The skill never
opens the DB just to figure out which session is current — it reads the file.

If the file is missing and a script needs a session, ``ensure_active_session``
silently creates one with default settings. Sessions are bookkeeping, not a
gate (SPEC §5.4).
"""

from __future__ import annotations

import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import apsw

from bartleby.db.connection import open_db
from bartleby.project import get_project_dir


# Agent harnesses leak their identity through environment variables. Each entry
# is (harness label, env var whose mere presence is an unambiguous signature).
# Order matters only if two harnesses ever collide — keep the most specific
# first. We deliberately recognize only signatures we can verify and return
# None for everything else: a wrong label is worse than an honest "unknown"
# (see issue #62). Add Goose / Pi / Ollama entries here as each is confirmed.
_HARNESS_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("claude-code", "CLAUDECODE"),
)


def detect_harness() -> str | None:
    """Best-effort harness identification from the environment, else None.

    Read at call time (not import) so it reflects the process actually
    spawning the skill, and so tests can control it via env.
    """
    for label, env_var in _HARNESS_SIGNATURES:
        if os.environ.get(env_var):
            return label
    return None


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
    model: str | None
    harness: str | None
    created_at: str
    ended_at: str | None
    run_key: str | None


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
    # Write to a *unique* sibling temp file, then atomically rename over the
    # pointer. A concurrent reader (read_active_session_id) either sees the old
    # id or the new one in full — never a half-written line. os.replace is atomic
    # on the same filesystem; the temp sits beside the target to stay there.
    #
    # The temp name must be unique per writer: a shared ".active_session.tmp" let
    # two concurrent writers clobber one temp and race on the rename — the second
    # os.replace then raised FileNotFoundError because the first already renamed
    # it away. Last writer wins, which is fine (all ids are valid). Clean up our
    # own temp on failure so a crash doesn't leave it behind.
    fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=f.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(session_id))
        os.replace(tmp, f)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def clear_active_session(project_name: str) -> None:
    f = _active_session_file(project_name)
    if f.exists():
        f.unlink()


def _row_to_info(row) -> SessionInfo:
    return SessionInfo(
        session_id=row[0],
        name=row[1],
        memory_enabled=bool(row[2]),
        model=row[3],
        harness=row[4],
        created_at=row[5],
        ended_at=row[6],
        run_key=row[7],
    )


def _fetch_session(conn, session_id: int) -> SessionInfo | None:
    row = conn.cursor().execute(
        "SELECT session_id, name, memory_enabled, model, harness, created_at, "
        "ended_at, run_key "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return _row_to_info(row) if row else None


def run_echo(conn, session_id: int) -> dict:
    """The current-run summary every skill result carries (#547).

    Echoed so the agent (and the human reading output) always sees which run a
    call landed in, and so the run_key stays fresh in the agent's context across
    a long, summarized conversation. ``model_set_by_llm`` is True only for an
    agent-minted run (run_key present) whose model the agent self-reported — it
    flags the model as a *claim* ("Set by LLM"), never a verified fact, and is
    always False for a CLI/web session. Returns ``{}`` if the row vanished.
    """
    info = _fetch_session(conn, session_id)
    if info is None:
        return {}
    return {
        "run_key": info["run_key"],
        "session_id": info["session_id"],
        "session_name": info["name"],
        "model": info["model"],
        "memory_enabled": info["memory_enabled"],
        "model_set_by_llm": bool(info["run_key"] and info["model"]),
    }


def start_session(
    project_name: str,
    *,
    memory_enabled: bool = True,
    model: str | None = None,
    harness: str | None = None,
    run_key: str | None = None,
    max_attempts: int = 32,
) -> SessionInfo:
    """Insert a new session, mark it active, return its info.

    Generates a memorable ``adjective-noun`` name; on collision (UNIQUE),
    retries with a fresh draw up to ``max_attempts`` times.

    ``model`` / ``harness`` record which backend authored the session's
    findings (issue #62). An explicit ``harness`` wins; when it's omitted we
    fall back to :func:`detect_harness`. ``model`` is rarely discoverable from
    the environment, so it stays NULL unless declared here or set later via
    :func:`set_session_provenance`.

    ``run_key`` is the per-conversation UUID an agent mints via ``skill session
    new`` (#547). The retry loop only redraws the *name*; the caller is
    responsible for supplying a fresh ``run_key`` (a uuid4 never collides), so a
    run_key UNIQUE violation is not something the redraw can clear — get-or-
    create against an existing key lives in :func:`ensure_session_by_run_key`.
    """
    harness = harness or detect_harness()
    conn = open_db(project_name)
    try:
        cur = conn.cursor()
        for _ in range(max_attempts):
            name = generate_name()
            try:
                cur.execute(
                    "INSERT INTO sessions (name, memory_enabled, model, harness, "
                    "run_key) VALUES (?, ?, ?, ?, ?)",
                    (name, 1 if memory_enabled else 0, model, harness, run_key),
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


def set_session_provenance(
    project_name: str,
    *,
    model: str | None = None,
    harness: str | None = None,
) -> SessionInfo | None:
    """Update the active session's ``model`` and/or ``harness`` in place.

    Only the fields you pass are written, so you can correct one without
    clobbering the other. Supports the blind-comparison workflow (start a
    session with provenance unset, stamp it after the fact) and fixing a wrong
    auto-detection. Returns the updated info, or ``None`` if no active session.
    """
    sid = read_active_session_id(project_name)
    if sid is None:
        return None
    conn = open_db(project_name)
    try:
        # COALESCE keeps a field untouched when its arg is None (mirrors the
        # "only write what you pass" contract); an UPDATE against a vanished
        # row is a harmless no-op, so no existence pre-check is needed.
        conn.cursor().execute(
            "UPDATE sessions "
            "SET model = COALESCE(?, model), harness = COALESCE(?, harness) "
            "WHERE session_id = ?",
            (model, harness, sid),
        )
        return _fetch_session(conn, sid)
    finally:
        conn.close()


def ensure_named_session(project_name: str, name: str) -> int:
    """Return the id of a durable, memory-enabled session with this exact
    ``name``, creating it if absent.

    Unlike :func:`ensure_active_session`, this never reads or writes the
    ``.active_session`` pointer. It exists for non-agent callers — chiefly the
    web UI — that need a stable session of their own without hijacking (or
    being hijacked by) whichever session an agent has active. The reserved
    name acts as the lookup key; reusing it across requests keeps every web
    invocation attributed to one recognizable, memory-enabled session.
    """
    conn = open_db(project_name)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT session_id FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row[0]
        try:
            cur.execute(
                "INSERT INTO sessions (name, memory_enabled) VALUES (?, 1)",
                (name,),
            )
            return conn.last_insert_rowid()
        except apsw.ConstraintError:
            # Concurrent caller won the insert race; re-read its row.
            row = cur.execute(
                "SELECT session_id FROM sessions WHERE name = ?", (name,)
            ).fetchone()
            return row[0]
    finally:
        conn.close()


def ensure_active_session(project_name: str) -> int:
    """Return the active session_id, creating one with defaults if missing.

    For use by skill scripts that must always have a session to attribute
    findings and audit-log entries to.
    """
    info = get_current_session(project_name) or start_session(
        project_name, memory_enabled=True
    )
    return info["session_id"]


def ensure_session_by_run_key(
    project_name: str,
    run_key: str,
    *,
    model: str | None = None,
    max_attempts: int = 32,
) -> int:
    """Return the session bound to ``run_key``, creating it if absent (#547).

    The per-conversation run resolver: an agent mints a run_key via ``skill
    session new`` and carries it on each call as ``--run <uuid>``. Get-or-create
    keyed on that UUID means a conversation's calls all resolve to one run, two
    conversations never collide (distinct keys), and a call that arrives before
    ``session new`` ran (or after it failed) still lands somewhere sane. It
    writes the ``.active_session`` marker so the no-``--run`` fallback
    (:func:`ensure_active_session`) keeps pointing at the most recent run.

    Race-safe: a pre-SELECT covers the common reuse case. On the create path the
    INSERT can trip either UNIQUE — ``name`` (redraw and retry) or ``run_key``
    (a concurrent caller minted the same key first, so re-SELECT and return the
    winner). Created runs are always memory-enabled; the memory-off eval path
    sets its policy explicitly via ``session new --no-memory``.
    """
    conn = open_db(project_name)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT session_id FROM sessions WHERE run_key = ?", (run_key,)
        ).fetchone()
        if row:
            write_active_session_id(project_name, row[0])
            return row[0]
        harness = detect_harness()
        for _ in range(max_attempts):
            name = generate_name()
            try:
                cur.execute(
                    "INSERT INTO sessions (name, memory_enabled, model, harness, "
                    "run_key) VALUES (?, 1, ?, ?, ?)",
                    (name, model, harness, run_key),
                )
            except apsw.ConstraintError:
                won = cur.execute(
                    "SELECT session_id FROM sessions WHERE run_key = ?", (run_key,)
                ).fetchone()
                if won:  # lost the run_key race — adopt the winner
                    write_active_session_id(project_name, won[0])
                    return won[0]
                continue  # a name collision — redraw and retry
            session_id = conn.last_insert_rowid()
            write_active_session_id(project_name, session_id)
            return session_id
        raise RuntimeError(
            f"Could not generate a unique session name after {max_attempts} tries; "
            f"the namespace may be exhausted."
        )
    finally:
        conn.close()
