"""`bartleby scribe backfill-dates` — bulk-populate ``authored_date`` from a
filename regex.

A **human-run admin op**, deliberately not a skill script: it's a bulk write
over a whole corpus (the press-release corpus has 174k docs with zero
summarizer-set dates but a date in every filename), kept off the agent surface
alongside ``scribe`` ingest and ``project upgrade``.

Per document, the regex's named ``date`` group is matched against ``file_name``
(basename) or, under ``--match-path``, the full ``file_path``. The captured raw
value is normalized via :func:`normalize_authored_date`; a match that fails
normalization is **counted and reported, never silently written as NULL**.

For a document that already has a summary row, this UPDATEs ``authored_date``
(only where it IS NULL, unless ``--overwrite``). For a document with no summary,
it INSERTs a date-only **stub**: ``model='backfill'``, empty title/description/
text, and **no summary chunks** (empty text is not chunked/embedded). Section
rows (#254) share their parent's ``file_name`` and so match the same regex and
inherit the parent's date — desired, so chunk/section-level date filtering
works; a stub is created for them too.

``--dry-run`` mutates nothing and prints the same counts plus a few sample
``file_name → date`` pairs. Idempotent / re-runnable.
"""

from __future__ import annotations

import sys

from bartleby.db.connection import open_db, resolve_project_name
from bartleby.ingest.summarize import (
    FilenameDateError,
    compile_filename_date_regex,
    extract_filename_date,
    normalize_authored_date,
)
from bartleby.lib import console
from bartleby.lib.consts import BACKFILL_MODEL

_SAMPLE_LIMIT = 10


def main(
    *,
    project: str | None,
    from_filename: str,
    match_path: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    try:
        compiled = compile_filename_date_regex(from_filename)
    except FilenameDateError as e:
        console.error(str(e))
        sys.exit(1)

    project_name = resolve_project_name(project)
    conn = open_db(project_name)
    try:
        cur = conn.cursor()
        # file_name + file_path + whether a (non-stub-or-not) summary row exists,
        # and its current authored_date, for every document. One pass.
        rows = cur.execute(
            "SELECT d.document_id, d.file_name, d.file_path, "
            "       s.summary_id, s.authored_date "
            "FROM documents d LEFT JOIN summaries s USING (document_id)"
        ).fetchall()

        matched = unmatched = invalid = 0
        to_update: list[tuple[int, str]] = []          # (summary_id, date)
        to_insert: list[tuple[int, str]] = []          # (document_id, date)
        samples: list[tuple[str, str]] = []            # (file_name, date)

        for document_id, file_name, file_path, summary_id, current_date in rows:
            subject = file_path if match_path else file_name
            raw = extract_filename_date(compiled, subject or "")
            if raw is None:
                unmatched += 1
                continue
            matched += 1
            value = normalize_authored_date(raw)
            if value is None:
                invalid += 1
                continue
            if len(samples) < _SAMPLE_LIMIT:
                samples.append((file_name, value))
            if summary_id is None:
                to_insert.append((document_id, value))
            elif current_date is None or overwrite:
                to_update.append((summary_id, value))
            # else: a dated summary row, no --overwrite → leave it (idempotent).

        if not dry_run:
            with conn:
                if to_update:
                    cur.executemany(
                        "UPDATE summaries SET authored_date = ? WHERE summary_id = ?",
                        [(value, sid) for sid, value in to_update],
                    )
                if to_insert:
                    # Stub: empty title/description/text, model='backfill', the
                    # date. NO summary chunks — empty text is not chunked, so the
                    # typed chunk helpers are correctly never called here.
                    cur.executemany(
                        "INSERT INTO summaries "
                        "(document_id, title, description, text, model, authored_date) "
                        "VALUES (?, '', '', '', ?, ?)",
                        [(document_id, BACKFILL_MODEL, value)
                         for document_id, value in to_insert],
                    )
    finally:
        conn.close()

    verb = "would " if dry_run else ""
    console.big(
        f"backfill-dates {'(dry run) ' if dry_run else ''}on '{project_name}'"
    )
    console.info(f"  matched:           {matched}")
    console.info(f"  unmatched:         {unmatched}")
    console.info(f"  invalid date:      {invalid}")
    console.info(f"  {verb}insert stub:  {len(to_insert)}")
    console.info(f"  {verb}update date:  {len(to_update)}")
    if samples:
        console.info("  samples:")
        for name, value in samples:
            console.info(f"    {name} → {value}")
    if not dry_run:
        console.complete("Done.")
