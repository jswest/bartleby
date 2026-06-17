#!/usr/bin/env python
"""Backfill legacy bare ``[^N]`` chunk-citation markers to ``[^chunk:N]``.

GH-0624 retyped the inline chunk-citation marker from bare ``[^N]`` to
``[^chunk:N]`` and pointed every reader (the web finding view, the
``bartleby finding export/import`` path) at the prefixed grammar — but storage
was left untouched and existing finding bodies were never migrated. So every
finding written before the cutover renders its chunk citations as inert text and
they no longer click through to the source pane (issue #642). A survey at the
time of writing found 2101 bare markers across 153 of 161 findings; only the
``test631`` project used the new form.

This one-shot maintenance script rewrites those bodies in place. It is **not** a
``bartleby <verb>`` command on purpose — it is a corpus-data fixup run once per
machine, not part of the day-to-day surface.

Only bare ``[^<digits>]`` markers are rewritten. Already-typed ``[^chunk:N]`` and
external ``[^url:...]`` / ``[^doc:...]`` markers never match, so the rewrite is
idempotent — running it twice changes nothing the second time. A bare ``[^N]``
was unambiguously a chunk citation before #624 (the stored ``finding_citations``
rows are the source of truth for *which* chunk), so the rewrite is lossless; a
marker whose id is no longer a live citation simply renders as a "no longer
available" note, exactly as the reader already handles.

Usage (from the repo root)::

    uv run python scripts/backfill_chunk_citations.py             # dry run, all projects
    uv run python scripts/backfill_chunk_citations.py --project northwestern
    uv run python scripts/backfill_chunk_citations.py --write      # apply (backs up each DB first)
    uv run python scripts/backfill_chunk_citations.py --write --no-backup
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bartleby.config import projects_dir  # noqa: E402
from bartleby.db.connection import project_db_path  # noqa: E402
from bartleby.project import validate_project_name  # noqa: E402

# Bare legacy chunk-citation marker: ``[^`` + all-digits + ``]``. The leading
# ``[^`` cannot be followed by a letter, so ``[^chunk:N]`` / ``[^url:...]`` /
# ``[^doc:...]`` are never touched — the rewrite is idempotent.
_BARE_MARKER = re.compile(r"\[\^(\d+)\]")
_BACKUP_SUFFIX = ".pre-chunk-backfill.bak"


def rewrite_body(body: str) -> tuple[str, int]:
    """Return ``(new_body, count)`` with bare ``[^N]`` rewritten to ``[^chunk:N]``."""
    new_body, count = _BARE_MARKER.subn(r"[^chunk:\1]", body)
    return new_body, count


def _project_dbs(only: str | None) -> list[Path]:
    if only:
        validate_project_name(only)  # rejects traversal / illegal names
        db = project_db_path(only)
        return [db] if db.exists() else []
    return sorted(projects_dir().glob("*/bartleby.db"))


def _backup(con: sqlite3.Connection, db: Path) -> None:
    """Snapshot ``db`` (pre-write) beside itself via SQLite's online backup.

    Uses ``Connection.backup`` rather than a file copy so the snapshot folds in
    any WAL state and is a consistent single file. An existing backup is never
    clobbered — the first one is the pristine pre-backfill copy.
    """
    bak = db.with_name(db.name + _BACKUP_SUFFIX)
    if bak.exists():
        print(f"  {db.parent.name:24s} backup exists, not overwriting", file=sys.stderr)
        return
    dest = sqlite3.connect(bak)
    try:
        with dest:
            con.backup(dest)
    finally:
        dest.close()


def _backfill_db(db: Path, *, write: bool, backup: bool) -> tuple[int, int]:
    """Rewrite one corpus. Return ``(findings_changed, markers_rewritten)``."""
    con = sqlite3.connect(db)
    con.execute("PRAGMA busy_timeout = 5000")  # ride out a transient writer lock
    try:
        edits = []
        for fid, body in con.execute("SELECT finding_id, body FROM findings"):
            new, n = rewrite_body(body or "")
            if n:
                edits.append((fid, new, n))
        markers = sum(n for _, _, n in edits)
        if not edits or not write:
            return len(edits), markers

        if backup:
            _backup(con, db)
        with con:  # one transaction per corpus — all-or-nothing
            con.executemany(
                "UPDATE findings SET body = ? WHERE finding_id = ?",
                [(new, fid) for fid, new, _ in edits],
            )
        return len(edits), markers
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill bare [^N] chunk-citation markers to [^chunk:N] in finding bodies."
    )
    parser.add_argument("--project", help="Only this project (default: every project corpus).")
    parser.add_argument(
        "--write", action="store_true",
        help="Apply the rewrite. Without it, this is a dry run.",
    )
    parser.add_argument(
        "--no-backup", dest="backup", action="store_false",
        help="Skip the per-DB <name>.pre-chunk-backfill.bak copy made before writing.",
    )
    args = parser.parse_args(argv)

    try:
        dbs = _project_dbs(args.project)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if not dbs:
        where = f"project {args.project!r}" if args.project else "any project"
        print(f"No corpus DB found for {where} under {projects_dir()}.", file=sys.stderr)
        return 1

    verb = "Rewriting" if args.write else "Would rewrite"
    total_findings = total_markers = 0
    for db in dbs:
        try:
            changed, markers = _backfill_db(db, write=args.write, backup=args.backup)
        except sqlite3.OperationalError as e:
            print(f"  {db.parent.name:24s} SKIPPED ({e})", file=sys.stderr)
            continue
        total_findings += changed
        total_markers += markers
        if changed:
            print(f"  {db.parent.name:24s} {verb.lower()} {markers} marker(s) in {changed} finding(s)")
        else:
            print(f"  {db.parent.name:24s} nothing to do")

    mode = "Rewrote" if args.write else "Dry run — would rewrite"
    print(f"\n{mode} {total_markers} marker(s) across {total_findings} finding(s) in {len(dbs)} corpus/corpora.")
    if not args.write and total_markers:
        print("Re-run with --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
