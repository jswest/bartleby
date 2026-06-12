"""Read-only corpus integrity checks for a Bartleby project DB.

The single home for the structural-consistency checks that both
``bartleby project info --verify`` (issue #487) and the test suite consume.
Nothing here mutates the corpus: every check is a read or a vacuous-write
integrity probe (FTS5's ``'integrity-check'`` command writes nothing durable).

Each check returns an :class:`IntegrityResult`; :func:`run_all_checks` runs the
full battery, and the caller collapses ``r.passed`` across results into an exit
code. The checks:

- **tri_table_sync** — ``chunks`` / ``chunks_fts`` / ``chunks_vec`` agree. The
  FTS leg uses FTS5's external-content ``'integrity-check'`` in the load-bearing
  ``rank=1`` form (a one-argument call is a no-op for content/index drift); the
  vec leg compares ``chunks_vec`` rowids against ``chunks.chunk_id`` exactly.
- **no_orphan_chunks** — every ``chunks.source_id`` resolves to a live parent
  row for its ``source_kind`` (document / summary / finding / image).
- **schema_matches_stamp** — the tables and indexes the stamped
  ``schema_version`` implies actually exist (catches missing-table corruption
  that would otherwise surface as a raw apsw traceback deep in a query).
- **failed_ingests_sanity** — ``failed_ingests`` rows are well-formed and don't
  contradict derived completeness (a ``parse``-stage failure must not coexist
  with live document chunks for the same file).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import apsw

from bartleby.db.schema import ALLOWED_SOURCE_KINDS, DDL, SCHEMA_VERSION

# The parent table + primary-key column backing each polymorphic source_kind.
# An orphaned chunk is one whose source_id has no row here.
_PARENT_OF_KIND = {
    "document": ("documents", "document_id"),
    "summary": ("summaries", "summary_id"),
    "finding": ("findings", "finding_id"),
    "image": ("images", "image_id"),
}


@dataclass
class IntegrityResult:
    """One check's outcome. ``passed`` drives the exit code; ``detail`` the line."""

    name: str
    passed: bool
    detail: str


def _expected_objects() -> set[tuple[str, str]]:
    """``(type, name)`` pairs the canonical DDL declares (tables + indexes).

    Used as a *subset* expectation against ``sqlite_master``: the FTS5 and vec0
    virtual tables spawn shadow tables not named in the DDL, so we require the
    declared objects to be present rather than demanding an exact match.
    """
    objects: set[tuple[str, str]] = set()
    for m in re.finditer(r"CREATE (?:VIRTUAL )?TABLE (\w+)", DDL):
        objects.add(("table", m.group(1)))
    for m in re.finditer(r"CREATE INDEX (\w+)", DDL):
        objects.add(("index", m.group(1)))
    return objects


def check_tri_table_sync(conn: apsw.Connection) -> IntegrityResult:
    """``chunks`` / ``chunks_fts`` / ``chunks_vec`` agree.

    FTS leg: external-content ``'integrity-check'`` in the ``rank=1`` form, which
    re-derives the index from ``chunks`` and raises on drift in either direction.
    Vec leg: ``chunks_vec`` rowid set compared exactly against ``chunks.chunk_id``.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)"
        )
    except apsw.Error as e:
        return IntegrityResult(
            "tri_table_sync", False, f"chunks_fts integrity-check failed: {e}"
        )

    chunk_ids = {r[0] for r in cur.execute("SELECT chunk_id FROM chunks")}
    vec_ids = {r[0] for r in cur.execute("SELECT rowid FROM chunks_vec")}
    if vec_ids != chunk_ids:
        only_vec = sorted(vec_ids - chunk_ids)
        only_chunks = sorted(chunk_ids - vec_ids)
        return IntegrityResult(
            "tri_table_sync",
            False,
            f"chunks_vec rowids drifted from chunks: "
            f"only in vec={only_vec}, only in chunks={only_chunks}",
        )
    return IntegrityResult(
        "tri_table_sync", True, f"fts + vec consistent across {len(chunk_ids)} chunks"
    )


def check_no_orphan_chunks(conn: apsw.Connection) -> IntegrityResult:
    """Every ``chunks.source_id`` resolves to a live parent row for its kind."""
    cur = conn.cursor()
    dangling: dict[str, int] = {}
    for kind in ALLOWED_SOURCE_KINDS:
        table, pk = _PARENT_OF_KIND[kind]
        count = cur.execute(
            f"SELECT COUNT(*) FROM chunks c "
            f"WHERE c.source_kind = ? "
            f"AND NOT EXISTS (SELECT 1 FROM {table} p WHERE p.{pk} = c.source_id)",
            (kind,),
        ).fetchone()[0]
        if count:
            dangling[kind] = count
    if dangling:
        parts = ", ".join(f"{k}={n}" for k, n in sorted(dangling.items()))
        return IntegrityResult(
            "no_orphan_chunks", False, f"orphaned chunks by kind: {parts}"
        )
    return IntegrityResult("no_orphan_chunks", True, "no orphaned chunks")


def check_schema_matches_stamp(conn: apsw.Connection) -> IntegrityResult:
    """Tables/indexes the stamped ``schema_version`` implies actually exist."""
    cur = conn.cursor()
    row = cur.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return IntegrityResult(
            "schema_matches_stamp", False, "no schema_version stamped in meta"
        )
    stamped = int(row[0])
    if stamped != SCHEMA_VERSION:
        return IntegrityResult(
            "schema_matches_stamp",
            False,
            f"stamped schema v{stamped} != code v{SCHEMA_VERSION}; "
            f"upgrade or re-ingest",
        )
    present = {
        (t, n)
        for t, n in cur.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index')"
        )
    }
    missing = sorted(_expected_objects() - present)
    if missing:
        parts = ", ".join(f"{t} {n}" for t, n in missing)
        return IntegrityResult(
            "schema_matches_stamp",
            False,
            f"schema v{stamped} stamped but objects missing: {parts}",
        )
    return IntegrityResult(
        "schema_matches_stamp", True, f"schema v{stamped} objects all present"
    )


def check_failed_ingests_sanity(conn: apsw.Connection) -> IntegrityResult:
    """``failed_ingests`` rows are well-formed and don't contradict completeness.

    A ``parse``-stage failure means the unit never produced text chunks, so it
    must not coexist with live document chunks for the same file. If it does, the
    failure record and the chunked document disagree about whether the unit
    landed — a corruption worth surfacing.
    """
    cur = conn.cursor()
    contradicted = cur.execute(
        "SELECT COUNT(*) FROM failed_ingests f "
        "WHERE f.stage = 'parse' AND EXISTS ("
        "  SELECT 1 FROM documents d "
        "  JOIN chunks c ON c.source_kind = 'document' AND c.source_id = d.document_id "
        "  WHERE d.file_hash = f.file_hash"
        ")"
    ).fetchone()[0]
    if contradicted:
        return IntegrityResult(
            "failed_ingests_sanity",
            False,
            f"{contradicted} parse-failure row(s) contradicted by live "
            f"document chunks for the same file",
        )
    total = cur.execute("SELECT COUNT(*) FROM failed_ingests").fetchone()[0]
    return IntegrityResult(
        "failed_ingests_sanity",
        True,
        f"{total} failed-ingest row(s), none contradicting completeness",
    )


_CHECKS = (
    check_tri_table_sync,
    check_no_orphan_chunks,
    check_schema_matches_stamp,
    check_failed_ingests_sanity,
)


def run_all_checks(conn: apsw.Connection) -> list[IntegrityResult]:
    """Run every integrity check against ``conn`` and return their results.

    Read-only: never opens a transaction, never mutates corpus rows.

    A check whose unguarded query hits a corrupt/missing object (e.g. a dropped
    ``chunks_vec``) raises ``apsw.Error``; we catch it per-check so it becomes a
    FAILED result instead of a raw traceback, and the remaining checks still run.
    """
    results: list[IntegrityResult] = []
    for check in _CHECKS:
        name = check.__name__.removeprefix("check_")
        try:
            results.append(check(conn))
        except apsw.Error as e:
            results.append(IntegrityResult(name, False, str(e)))
    return results
