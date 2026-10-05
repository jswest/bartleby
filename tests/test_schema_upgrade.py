"""Schema/upgrade invariants: the upgrade chain, mid-chain crash resume, refusing
a DB newer than the code, and the release schema-drift check.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import re

import pytest

from bartleby.db.chunks import ChunkInput
from bartleby.db.chunks import insert_finding_chunks
from bartleby.db.connection import open_db
from bartleby.db.schema import SCHEMA_VERSION
import bartleby.config
import bartleby.project

from tests._skill_fixtures import _emb


# ---------- project ----------


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
        # v11 finding_annotations (#689): a new table + index, dropped whole.
        cur.execute("DROP INDEX idx_finding_annotations_finding")
        cur.execute("DROP TABLE finding_annotations")
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


def test_upgrade_chain_walks_from_v4_through_current(projects_root):
    """Upgrading a v4 DB walks v4→v5→…→v11, leaving all new shapes present.

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


# ---------- release ----------


_RELEASE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "release.py"


_spec = importlib.util.spec_from_file_location("release", _RELEASE_PATH)


release = importlib.util.module_from_spec(_spec)


_spec.loader.exec_module(release)


SCHEMA_TEMPLATE = '''\
"""schema."""

SCHEMA_VERSION = {version}

EMBEDDING_DIM = 768

DDL = """{ddl}"""
'''


def _schema_source(version: int, ddl: str = "CREATE TABLE meta (k TEXT);") -> str:
    return SCHEMA_TEMPLATE.format(version=version, ddl=ddl)


@pytest.mark.parametrize(
    ("old", "new", "drifts"),
    [
        pytest.param(_schema_source(7), _schema_source(7), False,
                     id="nothing_changed"),
        pytest.param(_schema_source(7, "CREATE TABLE a (x);"),
                     _schema_source(8, "CREATE TABLE a (x); CREATE TABLE b (y);"),
                     False, id="ddl_changed_and_version_bumped"),
        pytest.param(_schema_source(7, "CREATE TABLE a (x);"),
                     _schema_source(7, "CREATE TABLE a (x); CREATE TABLE b (y);"),
                     True, id="ddl_changed_but_version_static"),
        pytest.param(_schema_source(7), _schema_source(8), False,
                     id="version_bump_without_ddl_change"),
    ],
)
def test_check_drift(old, new, drifts):
    problem = release.check_drift(old, new)
    if drifts:
        assert problem is not None
        assert "SCHEMA_VERSION" in problem
    else:
        assert problem is None
