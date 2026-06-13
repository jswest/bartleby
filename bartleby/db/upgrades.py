"""Additive-only schema upgrades.

The default position is "no backwards compat" — bumps mean re-ingest. This
chain is the one allowed relaxation: when a bump is *purely additive* (new
tables, indexes, nullable columns), a function here lets existing users run
``bartleby project upgrade <name>`` instead. Non-additive bumps simply have
no entry; ``project upgrade`` refuses.

The codebase never branches on schema version — ``SCHEMA_VERSION`` stays
pinned. This chain is the one-shot gate.

Note: the DDL here intentionally duplicates `db/schema.py` — fresh DBs run
the latter, existing DBs walk the former. The regression gate that keeps
them in sync is `tests/test_project.py::test_upgrade_chain_walks_*`, which
strips and re-applies the chain end-to-end.

The #164–#171 window cohort (v8 in `meta` but missing `ingests`/`ingest_run_id`)
is re-ingest-only and never a target of this chain — see `docs/decisions/GH-0352`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import apsw

from bartleby.db.schema import SCHEMA_VERSION
from bartleby.lib.consts import EMBEDDING_MODEL


def _upgrade_v4_to_v5(conn: apsw.Connection) -> None:
    conn.cursor().execute(
        "ALTER TABLE summaries ADD COLUMN authored_date TEXT"
    )


def _upgrade_v5_to_v6(conn: apsw.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE tags ("
        "  tag_id INTEGER PRIMARY KEY, "
        "  name TEXT NOT NULL UNIQUE, "
        "  description TEXT NOT NULL, "
        "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    cur.execute(
        "CREATE TABLE document_tags ("
        "  document_id INTEGER NOT NULL "
        "    REFERENCES documents(document_id) ON DELETE CASCADE, "
        "  tag_id INTEGER NOT NULL "
        "    REFERENCES tags(tag_id) ON DELETE CASCADE, "
        "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "  PRIMARY KEY (document_id, tag_id)"
        ")"
    )
    cur.execute("CREATE INDEX idx_document_tags_tag ON document_tags(tag_id)")


def _upgrade_v6_to_v7(conn: apsw.Connection) -> None:
    cur = conn.cursor()
    cur.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
    cur.execute("ALTER TABLE sessions ADD COLUMN harness TEXT")


def _upgrade_v7_to_v8(conn: apsw.Connection) -> None:
    # Schema v8 is the resumable/provenance bump (#164 + #171): a
    # failed_ingests retry ledger, an ingests provenance table, and an
    # ingest_run_id stamp on every per-unit table. All purely additive — no
    # reingest. Keep this DDL in lockstep with db/schema.py.
    cur = conn.cursor()
    # failed_ingests records a per-unit ingest failure (parse / caption /
    # summary) so resumable ingest can cap retries on a deterministically
    # failing unit instead of attempting it every run.
    cur.execute(
        "CREATE TABLE failed_ingests ("
        "  file_hash TEXT NOT NULL, "
        "  file_name TEXT NOT NULL, "
        "  stage TEXT NOT NULL "
        "    CHECK (stage IN ('parse', 'caption', 'summary')), "
        "  error TEXT NOT NULL, "
        "  attempts INTEGER NOT NULL DEFAULT 1, "
        "  last_attempt TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "  PRIMARY KEY (file_hash, stage)"
        ")"
    )
    # ingests provenance table (mirrors schema.py). Old rows pre-date any run,
    # so their ingest_run_id stays NULL — which the chunk helpers expect.
    cur.execute(
        "CREATE TABLE ingests ("
        "  run_id INTEGER PRIMARY KEY, "
        "  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "  finished_at TEXT, "
        "  config_json TEXT NOT NULL, "
        "  bartleby_version TEXT, "
        "  schema_version INTEGER NOT NULL"
        ")"
    )
    # per-unit provenance columns (mirror schema.py). Nullable with no
    # default, so the FK-bearing ADD COLUMN is legal on a populated table.
    for table in ("documents", "summaries", "chunks"):
        cur.execute(
            f"ALTER TABLE {table} ADD COLUMN "
            "ingest_run_id INTEGER REFERENCES ingests(run_id)"
        )


def _upgrade_v8_to_v9(conn: apsw.Connection) -> None:
    # Schema v9 is the value-bearing-tags bump (#114): tags gain an optional
    # per-document value, produced by a stored regex run over chunks. All
    # purely additive — nullable columns, NULL truthful on pre-upgrade rows
    # (an ordinary boolean tag leaves them all NULL), so existing corpora run
    # `bartleby project upgrade` rather than re-ingest. Keep this DDL in
    # lockstep with db/schema.py.
    cur = conn.cursor()
    # tags: the value-tag method. value_type is the discriminator (NULL = an
    # ordinary boolean category tag); pattern is the extraction regex.
    cur.execute(
        "ALTER TABLE tags ADD COLUMN value_type TEXT "
        "CHECK (value_type IN ('number', 'string', 'date'))"
    )
    cur.execute("ALTER TABLE tags ADD COLUMN pattern TEXT")
    # document_tags: the extracted value + its chunk anchor (the citation
    # source). Nullable with no default, so the FK-bearing ADD COLUMN is legal
    # on a populated table; old rows keep value/chunk_id NULL.
    cur.execute("ALTER TABLE document_tags ADD COLUMN value TEXT")
    cur.execute(
        "ALTER TABLE document_tags ADD COLUMN "
        "chunk_id INTEGER REFERENCES chunks(chunk_id)"
    )
    # #254 anchor-splitting columns on `documents`. Additive: an existing corpus
    # keeps every document's parent_document_id/anchor_id/section_title/
    # section_order NULL (a truthful "whole, unsplit file"), so it upgrades in
    # place — sectioning old monoliths is a voluntary re-ingest, not forced.
    # Nullable with no default, so the self-referential FK ADD COLUMN is legal
    # on a populated table. Keep this DDL in lockstep with db/schema.py.
    cur.execute(
        "ALTER TABLE documents ADD COLUMN "
        "parent_document_id INTEGER REFERENCES documents(document_id)"
    )
    cur.execute("ALTER TABLE documents ADD COLUMN anchor_id TEXT")
    cur.execute("ALTER TABLE documents ADD COLUMN section_title TEXT")
    cur.execute("ALTER TABLE documents ADD COLUMN section_order INTEGER")


def _upgrade_v9_to_v10(conn: apsw.Connection) -> None:
    # Schema v10 binds one research run to one conversation via an agent-minted
    # UUID (#547): `sessions` gains a nullable `run_key` plus a UNIQUE index over
    # it. Purely additive — every pre-upgrade session keeps run_key NULL (a
    # truthful "this run was never UUID-minted"), and the unique index permits
    # those many NULLs while forbidding two sessions from sharing a key. So
    # existing corpora run `bartleby project upgrade` rather than re-ingest. Keep
    # this DDL in lockstep with db/schema.py.
    cur = conn.cursor()
    cur.execute("ALTER TABLE sessions ADD COLUMN run_key TEXT")
    cur.execute("CREATE UNIQUE INDEX idx_sessions_run_key ON sessions(run_key)")


_UPGRADES: dict[int, Callable[[apsw.Connection], None]] = {
    4: _upgrade_v4_to_v5,
    5: _upgrade_v5_to_v6,
    6: _upgrade_v6_to_v7,
    7: _upgrade_v7_to_v8,
    8: _upgrade_v8_to_v9,
    9: _upgrade_v9_to_v10,
}


def upgrade(conn: apsw.Connection, current_version: int) -> None:
    """Walk the chain from ``current_version`` up to ``SCHEMA_VERSION``.

    Raises if a step is missing — non-additive bumps have no entry and force
    re-ingest. Also raises, without stamping or mutating anything, if the DB is
    newer than this code: the loop would never run and the unconditional
    `upgraded_at` write below would otherwise rewrite a newer DB this code can't
    reason about. Keep the wording in lockstep with the CLI guard in
    `commands/project.py`.
    """
    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"DB schema v{current_version} is newer than this code "
            f"(v{SCHEMA_VERSION}). Update the code, not the DB."
        )
    v = current_version
    while v < SCHEMA_VERSION:
        step = _UPGRADES.get(v)
        if step is None:
            raise RuntimeError(
                f"No additive upgrade from v{v} to v{v + 1}. "
                f"This is a non-additive bump; re-ingest is required."
            )
        # Stamp the new version INSIDE the step's transaction: the DDL and the
        # schema_version bump commit together. A crash between two steps leaves
        # the DB at the last completed step's version (never a structural-vs-
        # meta mismatch), so a re-run resumes from there and completes.
        with conn:
            step(conn)
            conn.cursor().execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                (str(v + 1),),
            )
        v += 1

    with conn:
        # Backfill the pinned embedding model for corpora created before init_db
        # started recording it (#517). INSERT OR IGNORE makes this idempotent and
        # additive: it never overwrites a value a corpus already carries, so it's
        # safe to re-run and needs no SCHEMA_VERSION bump. Import-side
        # verification of the recorded value lives in #520, not here.
        conn.cursor().execute(
            "INSERT OR IGNORE INTO meta (key, value) "
            "VALUES ('embedding_model', ?)",
            (EMBEDDING_MODEL,),
        )
        conn.cursor().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('upgraded_at', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
