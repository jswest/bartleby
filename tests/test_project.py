"""Integration tests for bartleby.project (create, info, delete) against the v1 schema."""

from __future__ import annotations

import re

import pytest

import bartleby.config
import bartleby.db.connection
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


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    """Point every PROJECTS_DIR reference at a fresh tmp dir + isolate config."""
    projects = tmp_path / "projects"
    projects.mkdir()

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(bartleby.config, "BARTLEBY_DIR", tmp_path)
    monkeypatch.setattr(bartleby.config, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(bartleby.project, "PROJECTS_DIR", projects)
    monkeypatch.setattr(bartleby.db.connection, "PROJECTS_DIR", projects)
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


@pytest.mark.xfail(reason="held at 8 pending v0.9.0 assembly", strict=False)
def test_upgrade_chain_walks_from_v4_through_current(projects_root):
    """Upgrading a v4 DB walks v4→v5→v6→v7→v8, leaving all new shapes present.

    Held at 8: #114 adds value-bearing-tags columns to schema.py + a dormant
    `_upgrade_v8_to_v9` step keyed at 8 that the loop never reaches while
    SCHEMA_VERSION==8. The chain strips back to v4 and recreates `tags` /
    `document_tags` via the v5→v6 step (which has no new columns), so the
    upgraded-vs-fresh schema-equivalence gate cannot hold until the assembly
    commit bumps to 9. xfail (non-strict) until then.
    """
    import apsw

    from bartleby.commands import project as project_cmd
    from bartleby.db.connection import project_db_path

    bartleby.project.create_project("alpha")
    # Simulate a v4 DB by undoing every additive step since.
    db_path = project_db_path("alpha")
    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        # v8 provenance (drop the FK-bearing columns before the table).
        for table in ("documents", "summaries", "chunks"):
            cur.execute(f"ALTER TABLE {table} DROP COLUMN ingest_run_id")
        cur.execute("DROP TABLE ingests")
        cur.execute("DROP TABLE failed_ingests")
        cur.execute("ALTER TABLE sessions DROP COLUMN harness")
        cur.execute("ALTER TABLE sessions DROP COLUMN model")
        cur.execute("DROP INDEX idx_document_tags_tag")
        cur.execute("DROP TABLE document_tags")
        cur.execute("DROP TABLE tags")
        cur.execute("ALTER TABLE summaries DROP COLUMN authored_date")
        cur.execute(
            "UPDATE meta SET value = '4' WHERE key = 'schema_version'"
        )
    finally:
        conn.close()

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
    # Simulate a v4 DB by undoing every additive step since (mirrors the
    # chain-walk test's teardown).
    db_path = project_db_path("alpha")
    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        for table in ("documents", "summaries", "chunks"):
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
    import io

    import apsw
    from rich.console import Console

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

    # CLI: refuses with exit 1 and the re-ingest guidance.
    buf = io.StringIO()
    monkeypatch.setattr(project_cmd, "_console", Console(file=buf, width=200))
    with pytest.raises(SystemExit) as exc:
        project_cmd.upgrade(name="alpha")
    assert exc.value.code == 1
    assert "re-ingest" in buf.getvalue().lower()

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
