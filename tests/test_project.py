"""Integration tests for bartleby.project (create, info, delete) against the v1 schema."""

from __future__ import annotations

import re

import pytest

import bartleby.config
import bartleby.project
from bartleby.db.chunks import (
    ChunkInput,
    insert_document_chunks,
    insert_finding_chunks,
)
from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM, SCHEMA_VERSION


def _emb(seed: float = 0.0) -> list[float]:
    return [seed + i * 0.001 for i in range(EMBEDDING_DIM)]


def _strip_db_to_v4(db_path) -> None:
    """Undo every additive step since v4 on the DB at ``db_path``, stamping it
    back to schema v4 so the upgrade chain can be re-walked end-to-end.

    Used by both upgrade-chain tests. Opens a raw connection (FK enforcement
    OFF, so dropping FK-referenced tables/columns is legal) and reverses the
    chain newest-first.
    """
    import apsw

    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        # v10 per-conversation run_key (#547): drop the unique index before the
        # column, so the re-walked v9→v10 step can re-ALTER + re-index cleanly.
        cur.execute("DROP INDEX idx_sessions_run_key")
        cur.execute("ALTER TABLE sessions DROP COLUMN run_key")
        # v9 value-bearing-tags (#114) + anchor-splitting (#254) columns: strip
        # them first so the re-walked v8→v9 step can re-ALTER them without a
        # `duplicate column name` (flagged by #355).
        cur.execute("ALTER TABLE tags DROP COLUMN value_type")
        cur.execute("ALTER TABLE tags DROP COLUMN pattern")
        cur.execute("ALTER TABLE document_tags DROP COLUMN value")
        cur.execute("ALTER TABLE document_tags DROP COLUMN chunk_id")
        # `documents` can't be reduced with DROP COLUMN here: schema.py annotates
        # the #254 columns with a multi-line `--` comment block, and SQLite's
        # DROP COLUMN re-parses the residual CREATE — once the annotated columns
        # are gone the dangling comment yields `incomplete input`. FK enforcement
        # is OFF on this raw connection and the table is empty, so rebuild it
        # straight to its v4 shape (no ingest_run_id, no #254 columns) instead.
        cur.execute("DROP TABLE documents")
        cur.execute(
            "CREATE TABLE documents ("
            "  document_id INTEGER PRIMARY KEY, "
            "  file_hash TEXT NOT NULL UNIQUE, "
            "  file_name TEXT NOT NULL, "
            "  file_path TEXT NOT NULL, "
            "  page_count INTEGER, "
            "  token_count INTEGER, "
            "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        # v8 provenance (drop the FK-bearing columns before the table).
        for table in ("summaries", "chunks"):
            cur.execute(f"ALTER TABLE {table} DROP COLUMN ingest_run_id")
        cur.execute("DROP TABLE ingests")
        cur.execute("DROP TABLE failed_ingests")
        cur.execute("ALTER TABLE sessions DROP COLUMN harness")
        cur.execute("ALTER TABLE sessions DROP COLUMN model")
        cur.execute("DROP INDEX idx_document_tags_tag")
        cur.execute("DROP TABLE document_tags")
        cur.execute("DROP TABLE tags")
        cur.execute("ALTER TABLE summaries DROP COLUMN authored_date")
        cur.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    finally:
        conn.close()


@pytest.fixture
def projects_root():
    """The per-test projects dir (isolated via conftest's _isolate_bartleby_home)."""
    projects = bartleby.config.projects_dir()
    projects.mkdir(parents=True, exist_ok=True)
    yield projects


def test_create_project_initializes_v1_schema(projects_root):
    bartleby.project.create_project("alpha")

    db_path = projects_root / "alpha" / "bartleby.db"
    archive = projects_root / "alpha" / "archive"
    assert db_path.exists()
    assert archive.is_dir()
    assert not (projects_root / "alpha" / "book").exists()
    assert not (projects_root / "alpha" / "memory").exists()

    conn = open_db("alpha")
    try:
        meta = dict(conn.cursor().execute("SELECT key, value FROM meta"))
        assert meta["schema_version"] == str(SCHEMA_VERSION)
    finally:
        conn.close()


def test_create_project_sets_active(projects_root):
    bartleby.project.create_project("alpha")
    assert bartleby.project.get_active_project() == "alpha"


def test_create_project_refuses_existing(projects_root):
    bartleby.project.create_project("alpha")
    with pytest.raises(FileExistsError):
        bartleby.project.create_project("alpha")


def test_project_info_reports_v1_stats(projects_root):
    bartleby.project.create_project("alpha")
    conn = open_db("alpha")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path) VALUES (?, ?, ?)",
            ("h", "doc.pdf", "/tmp/doc.pdf"),
        )
        doc_id = conn.last_insert_rowid()
        insert_document_chunks(conn, doc_id, [
            ChunkInput(text="a", embedding=_emb(), chunk_index=0),
            ChunkInput(text="b", embedding=_emb(), chunk_index=1),
        ])
        cur.execute("INSERT INTO sessions (name) VALUES (?)", ("test-sess",))
        sid = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO findings (session_id, title, description, body) "
            "VALUES (?, ?, ?, ?)",
            (sid, "t", "d", "b"),
        )
    finally:
        conn.close()

    info = bartleby.project.get_project_info("alpha")
    assert info["schema_version"] == str(SCHEMA_VERSION)
    assert info["embedding_model"]
    assert info["document_count"] == 1
    assert info["session_count"] == 1
    assert info["finding_count"] == 1
    assert info["chunk_counts"] == {
        "document": 2, "summary": 0, "finding": 0, "image": 0,
    }


def _seed_failed_ingests(name: str, rows: list[tuple[str, str, str, str, int]]) -> None:
    """Insert (file_hash, file_name, stage, error, attempts) rows into failed_ingests."""
    conn = open_db(name)
    try:
        conn.cursor().executemany(
            "INSERT INTO failed_ingests "
            "(file_hash, file_name, stage, error, attempts, last_attempt) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            rows,
        )
    finally:
        conn.close()


def test_project_info_reports_failed_ingests(projects_root):
    """get_project_info counts every failed unit and, separately, the capped ones."""
    from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS

    bartleby.project.create_project("alpha")
    # No failures yet → both counters sit at zero.
    assert bartleby.project.get_project_info("alpha")["failed_ingests"] == {
        "total": 0, "capped": 0,
    }

    _seed_failed_ingests("alpha", [
        ("h1", "capped.pdf", "parse", "boom", MAX_INGEST_ATTEMPTS),
        ("h2", "over.pdf", "caption", "boom", MAX_INGEST_ATTEMPTS + 2),
        ("h3", "retry.pdf", "summary", "boom", 1),
    ])

    failed = bartleby.project.get_project_info("alpha")["failed_ingests"]
    assert failed == {"total": 3, "capped": 2}


def test_project_info_command_renders_failed_units_row(projects_root, monkeypatch):
    """`bartleby project info` renders the Failed units row only when units failed."""
    import io

    from rich.console import Console

    from bartleby.commands import project as project_cmd
    from bartleby.ingest.writer import MAX_INGEST_ATTEMPTS

    bartleby.project.create_project("alpha")

    def _render() -> str:
        buf = io.StringIO()
        monkeypatch.setattr(project_cmd, "_console", Console(file=buf, width=200))
        project_cmd.info(name="alpha")
        return buf.getvalue()

    # No failures → no row.
    assert "Failed units" not in _render()

    _seed_failed_ingests("alpha", [
        ("h1", "capped.pdf", "parse", "boom", MAX_INGEST_ATTEMPTS),
        ("h2", "retry.pdf", "summary", "boom", 1),
    ])

    out = _render()
    assert "Failed units" in out
    assert "2 incomplete" in out
    assert "1 capped, not retried" in out


def test_resolve_project_name_rejects_traversal_before_path(projects_root):
    """An explicit traversal `--project` is rejected at the resolve chokepoint.

    `resolve_project_name` is the single funnel for `open_db`, the skill runner,
    scribe, session, and logs. Validating here means a name like `../../x` raises
    the invalid-name `ValueError` before any `PROJECTS_DIR / name` path is built,
    mkdir runs, or a DB is opened — so no file lands at the traversed location.
    """
    from bartleby.db.connection import resolve_project_name

    with pytest.raises(ValueError, match="Invalid project name"):
        resolve_project_name("../../x")

    # Nothing was created outside the projects root.
    assert not (projects_root.parent / "x").exists()
    # A valid name resolves through untouched (positive control).
    assert resolve_project_name("alpha") == "alpha"


def test_open_db_rejects_traversal_name(projects_root):
    """`open_db` rejects a traversal name before touching the filesystem.

    Covers the skill runner / session / scribe paths, which all reach the DB via
    `open_db` → `resolve_project_name`.
    """
    from bartleby.db.connection import open_db

    with pytest.raises(ValueError, match="Invalid project name"):
        open_db("../../etc/passwd")


def test_set_active_project_rejects_traversal_before_path(projects_root):
    """`bartleby project use ../x` is rejected before the dir is even probed.

    `set_active_project` used to only check `project_dir.exists()`, persisting a
    traversal name into the active-project config. Validation up front raises the
    invalid-name `ValueError` first, and the config is never written.
    """
    before = bartleby.project.get_active_project()
    with pytest.raises(ValueError, match="Invalid project name"):
        bartleby.project.set_active_project("../x")
    assert bartleby.project.get_active_project() == before  # unchanged

    # A valid, existing project still switches (positive control).
    bartleby.project.create_project("alpha")
    bartleby.project.create_project("beta")
    bartleby.project.set_active_project("alpha")
    assert bartleby.project.get_active_project() == "alpha"


def test_delete_project_removes_dir_and_clears_active(projects_root):
    bartleby.project.create_project("alpha")
    bartleby.project.delete_project("alpha")
    assert not (projects_root / "alpha").exists()
    assert bartleby.project.get_active_project() is None


def test_list_projects_marks_active(projects_root):
    bartleby.project.create_project("alpha")
    bartleby.project.create_project("beta")
    listing = bartleby.project.list_projects()
    by_name = {p["name"]: p for p in listing}
    assert by_name["alpha"]["has_db"] and not by_name["alpha"]["is_active"]
    assert by_name["beta"]["has_db"] and by_name["beta"]["is_active"]


def test_upgrade_chain_walks_from_v4_through_current(projects_root):
    """Upgrading a v4 DB walks v4→v5→v6→v7→v8→v9, leaving all new shapes present.

    The v0.9.0 assembly bumped SCHEMA_VERSION to 9, activating the additive
    `_upgrade_v8_to_v9` step (#114 value-bearing-tags + #254 anchor-splitting
    columns). The chain strips a fresh DB back to v4 — including the eight v9
    columns — then re-walks the whole chain and asserts the upgraded DB is
    byte-identical in DDL to a freshly-created one.
    """
    from bartleby.commands import project as project_cmd
    from bartleby.db.connection import project_db_path

    bartleby.project.create_project("alpha")
    # Simulate a v4 DB by undoing every additive step since.
    db_path = project_db_path("alpha")
    _strip_db_to_v4(db_path)

    project_cmd.upgrade(name="alpha")

    conn = open_db("alpha")
    try:
        cur = conn.cursor()
        meta = dict(cur.execute("SELECT key, value FROM meta"))
        assert meta["schema_version"] == str(SCHEMA_VERSION)
        # v5 column landed.
        cols = [
            row[1] for row in cur.execute("PRAGMA table_info(summaries)")
        ]
        assert "authored_date" in cols
        # v6 tables landed.
        names = {
            row[0] for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "tags" in names
        assert "document_tags" in names
        # v7 columns landed.
        session_cols = [
            row[1] for row in cur.execute("PRAGMA table_info(sessions)")
        ]
        assert "model" in session_cols
        assert "harness" in session_cols
        # v8 retry ledger + provenance landed (the half-migration this fixes:
        # the upgrade used to stamp v8 but omit ingests + ingest_run_id).
        assert "failed_ingests" in names
        assert "ingests" in names
        for table in ("documents", "summaries", "chunks"):
            tcols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]
            assert "ingest_run_id" in tcols, table
    finally:
        conn.close()

    # The upgraded DB must be schema-equivalent to a freshly-created DB down to
    # the full DDL: same tables AND indexes, with identical column types,
    # NOT NULL / DEFAULT / CHECK constraints, and FK clauses. This is the
    # regression gate that keeps the upgrade chain in lockstep with
    # db/schema.py — comparing only table+column names (the old check) let
    # dropped indexes, type drift, and constraint/FK drift slip through.
    bartleby.project.create_project("beta")

    def _top_level_defs(body: str) -> tuple[str, ...]:
        """Split a CREATE TABLE body on its top-level commas, normalized + sorted.

        Each piece is one column or table-constraint definition (carrying its
        type, ``NOT NULL`` / ``DEFAULT`` / ``CHECK``, and ``REFERENCES`` FK
        clause). Splitting only at paren-depth 0 keeps a ``CHECK (... IN (...))``
        or a multi-column ``PRIMARY KEY (a, b)`` intact. Sorting makes the set
        order-independent: the chain's ``ALTER TABLE ADD COLUMN`` appends, so a
        chain-built table lists the same columns in a different order than
        schema.py — equivalent schemas we must not flag.
        """
        parts, depth, current = [], 0, ""
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        parts.append(current)
        return tuple(sorted(" ".join(p.split()) for p in parts if p.strip()))

    def _normalize(sql: str):
        # Strip `--` line comments (schema.py annotates columns; the chain's
        # hand-built CREATEs don't — pure layout, not structural drift) and
        # collapse whitespace, so only real DDL differences survive.
        sql = re.sub(r"--[^\n]*", "", sql)
        sql = " ".join(sql.split())
        if sql.upper().startswith("CREATE TABLE"):
            open_paren, close_paren = sql.find("("), sql.rfind(")")
            head = sql[:open_paren].strip().upper()
            return (head, _top_level_defs(sql[open_paren + 1 : close_paren]))
        # Indexes (and any non-plain-table CREATE) compare whole: a dropped or
        # altered index changes this normalized string outright.
        return (sql.upper(),)

    def _schema(conn):
        # Compare the full DDL of every table and index — column types,
        # NOT NULL / DEFAULT / CHECK constraints, FK clauses, and index
        # definitions — between the chain-upgraded DB and a fresh one, after
        # normalizing away whitespace, comments, and (within a table) column
        # order so pure formatting never false-positives. Rows with NULL sql
        # (autoindexes, the FTS5 / vec0 shadow internals) carry no
        # author-written DDL to diff and are skipped; both DBs build those
        # identically from the same CREATEs.
        return {
            (kind, name): _normalize(sql)
            for kind, name, sql in conn.cursor().execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'index') AND sql IS NOT NULL"
            )
        }

    upgraded, fresh = open_db("alpha"), open_db("beta")
    try:
        assert _schema(upgraded) == _schema(fresh)
    finally:
        upgraded.close()
        fresh.close()

    # The crash this fixes: chunk inserts unconditionally name ingest_run_id,
    # so saving a finding on the upgraded DB used to raise "no such column".
    conn = open_db("alpha")
    try:
        insert_finding_chunks(conn, 1, [
            ChunkInput(text="finding body", embedding=_emb(), chunk_index=0),
        ])
        count = conn.cursor().execute(
            "SELECT count(*) FROM chunks WHERE source_kind = 'finding'"
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_upgrade_resumes_after_mid_chain_crash(projects_root, monkeypatch):
    """A crash mid-chain leaves the DB at the last completed step; a re-run finishes.

    Each `_upgrade_vN_to_vN+1` now stamps `meta.schema_version` inside its own
    `with conn:`, so a step that raises after earlier steps committed leaves the
    DB at an intermediate version (not a structural-vs-meta mismatch). A re-run
    resumes from there and completes to SCHEMA_VERSION with no double-application
    (re-walking a committed step would raise "table already exists").
    """
    import apsw

    from bartleby.commands import project as project_cmd
    from bartleby.db import upgrades as upgrades_mod
    from bartleby.db.connection import project_db_path

    bartleby.project.create_project("alpha")
    # Simulate a v4 DB by undoing every additive step since.
    db_path = project_db_path("alpha")
    _strip_db_to_v4(db_path)

    # Kill the chain at the v6→v7 step: v4→v5 and v5→v6 commit first, then this
    # raises. The DB must come to rest at v6 (the last completed step), with the
    # v5/v6 shapes present and the v7 shape absent.
    real_v6_to_v7 = upgrades_mod._UPGRADES[6]
    crashed = {"hit": False}

    def boom(_conn):
        crashed["hit"] = True
        raise apsw.SQLError("simulated mid-chain crash")

    monkeypatch.setitem(upgrades_mod._UPGRADES, 6, boom)

    with pytest.raises(apsw.Error):
        upgrades_mod.upgrade(apsw.Connection(str(db_path)), 4)
    assert crashed["hit"]

    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        meta = dict(cur.execute("SELECT key, value FROM meta"))
        # Stamped to the last completed step, not the original v4 or the target.
        assert meta["schema_version"] == "6"
        names = {
            row[0] for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "tags" in names and "document_tags" in names  # v6 landed
        scols = [r[1] for r in cur.execute("PRAGMA table_info(sessions)")]
        assert "model" not in scols  # v7 did NOT land
    finally:
        conn.close()

    # Re-run with the real step restored: it must resume from v6 (never
    # re-applying v5/v6 — that would raise "table already exists") and complete.
    monkeypatch.setitem(upgrades_mod._UPGRADES, 6, real_v6_to_v7)
    project_cmd.upgrade(name="alpha")

    conn = open_db("alpha")
    try:
        cur = conn.cursor()
        meta = dict(cur.execute("SELECT key, value FROM meta"))
        assert meta["schema_version"] == str(SCHEMA_VERSION)
        scols = [r[1] for r in cur.execute("PRAGMA table_info(sessions)")]
        assert "model" in scols and "harness" in scols  # v7 now landed
        names = {
            row[0] for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "ingests" in names and "failed_ingests" in names  # v8 landed
    finally:
        conn.close()


def test_upgrade_refuses_below_chain_first_step_without_mutation(
    projects_root, monkeypatch
):
    """A DB below the chain's first entry is refused, leaving the DB unchanged.

    The chain's first step is keyed at v4 (`_upgrade_v4_to_v5`), so a DB stamped
    at v3 has no step to walk from: `upgrade()`'s loop hits `_UPGRADES.get(3) is
    None` and raises ("non-additive bump; re-ingest is required"), and the CLI
    wrapper surfaces that as exit 1. This pins the refusal property the
    per-step-commit structure makes load-bearing — a refused upgrade must not
    persist any DDL or move the version stamp. We assert both the full
    `sqlite_master` DDL and the `schema_version` stamp are byte-identical
    before and after.
    """
    import apsw

    from bartleby.commands import project as project_cmd
    from bartleby.db import upgrades as upgrades_mod
    from bartleby.db.connection import project_db_path

    bartleby.project.create_project("alpha")
    db_path = project_db_path("alpha")

    # Stamp the DB at v3 — one below the chain's first entry (v4). No code
    # branches on schema version, so the stale stamp alone reproduces the
    # "below the chain" condition; the structural DDL stays at the fresh shape.
    below_first = min(upgrades_mod._UPGRADES) - 1
    conn = apsw.Connection(str(db_path))
    try:
        conn.cursor().execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(below_first),),
        )
    finally:
        conn.close()

    def _snapshot():
        conn = apsw.Connection(str(db_path))
        try:
            cur = conn.cursor()
            master = cur.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name"
            ).fetchall()
            stamp = cur.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            return master, stamp
        finally:
            conn.close()

    before = _snapshot()
    assert before[1] == str(below_first)

    # CLI: refuses with exit 1 and the re-ingest guidance, routed through
    # console.error (stderr) per #491.
    errors: list[str] = []
    monkeypatch.setattr(project_cmd.console, "error", lambda m: errors.append(m))
    with pytest.raises(SystemExit) as exc:
        project_cmd.upgrade(name="alpha")
    assert exc.value.code == 1
    assert any("re-ingest" in m.lower() for m in errors)

    # The refusal mutated nothing: full DDL and the version stamp are identical.
    after = _snapshot()
    assert after == before


def test_upgrade_refuses_db_newer_than_code(projects_root):
    """A DB stamped newer than the code is refused without mutation.

    With `current_version > SCHEMA_VERSION` the chain loop never runs, so the
    unconditional `upgraded_at` write at the tail used to rewrite a newer DB's
    `schema_version` *down* to the code's — silently corrupting the version
    contract. `upgrade()` must raise before touching anything, and the CLI
    wrapper must surface that as exit 1 ("Update the code, not the DB").
    """
    import apsw

    from bartleby.commands import project as project_cmd
    from bartleby.db import upgrades as upgrades_mod
    from bartleby.db.connection import project_db_path

    bartleby.project.create_project("alpha")
    db_path = project_db_path("alpha")

    # Stamp the DB one version ahead of the code.
    newer = SCHEMA_VERSION + 1
    conn = apsw.Connection(str(db_path))
    try:
        conn.cursor().execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(newer),)
        )
        before = dict(conn.cursor().execute("SELECT key, value FROM meta"))
    finally:
        conn.close()

    # Library: raises, and stamps/mutates nothing.
    with pytest.raises(RuntimeError, match="newer than this code"):
        upgrades_mod.upgrade(apsw.Connection(str(db_path)), newer)

    conn = apsw.Connection(str(db_path))
    try:
        after = dict(conn.cursor().execute("SELECT key, value FROM meta"))
        assert after["schema_version"] == str(newer)  # not rewritten down
        assert "upgraded_at" not in after  # tail write never ran
        assert after == before  # nothing mutated at all
    finally:
        conn.close()

    # CLI: refuses with exit 1.
    with pytest.raises(SystemExit) as exc:
        project_cmd.upgrade(name="alpha")
    assert exc.value.code == 1


def test_upgrade_backfills_embedding_model_on_current_v9(projects_root):
    """`project upgrade` on an already-v9 corpus still backfills embedding_model.

    A corpus created before #517 is at schema v9 but carries no
    `meta.embedding_model` key, so `project publish` produces an artifact every
    importer hard-refuses. The CLI `upgrade` handler used to `return` early when
    `current == SCHEMA_VERSION`, BEFORE calling `upgrades.upgrade()` — so the
    always-runs backfill tail never ran and those corpora could never be fixed
    through the CLI. This goes through the public command path (not
    `upgrades.upgrade` directly, which the unit tests already cover) to prove the
    seam is reachable, and that a second run is idempotent.
    """
    import apsw

    from bartleby.commands import project as project_cmd
    from bartleby.db.connection import project_db_path
    from bartleby.lib.consts import EMBEDDING_MODEL

    bartleby.project.create_project("alpha")
    db_path = project_db_path("alpha")

    # Simulate a pre-#517 v9 corpus: at SCHEMA_VERSION but no embedding_model key.
    conn = apsw.Connection(str(db_path))
    try:
        conn.cursor().execute("DELETE FROM meta WHERE key = 'embedding_model'")
        meta = dict(conn.cursor().execute("SELECT key, value FROM meta"))
        assert meta["schema_version"] == str(SCHEMA_VERSION)
        assert "embedding_model" not in meta
    finally:
        conn.close()

    # CLI upgrade on an already-current DB now backfills the key.
    project_cmd.upgrade(name="alpha")

    conn = apsw.Connection(str(db_path))
    try:
        meta = dict(conn.cursor().execute("SELECT key, value FROM meta"))
        assert meta["embedding_model"] == EMBEDDING_MODEL
        assert meta["schema_version"] == str(SCHEMA_VERSION)
        first_upgraded_at = meta.get("upgraded_at")
    finally:
        conn.close()

    # Idempotent: a second run neither errors nor overwrites the existing value.
    project_cmd.upgrade(name="alpha")
    conn = apsw.Connection(str(db_path))
    try:
        meta = dict(conn.cursor().execute("SELECT key, value FROM meta"))
        assert meta["embedding_model"] == EMBEDDING_MODEL
    finally:
        conn.close()
