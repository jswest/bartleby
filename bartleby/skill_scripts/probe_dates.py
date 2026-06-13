#!/usr/bin/env python3
r"""probe_dates — dry-run a date regex over the corpus before prompting a human.

A **read-only** validation surface (writes nothing). Given a regex with a named
``date`` group, it reports how well that regex would extract dates from each
document's ``file_name`` (default) or a sample of document body text — so an
agent can *confirm* a candidate regex works before telling a human to run the
human-only admin command ``bartleby scribe backfill-dates``. The agent cannot
run the backfill itself; this script is the agent-side discovery/validation
twin of the backfill command's own ``--dry-run`` preview.

It reuses the backfill helpers (#536): ``compile_filename_date_regex`` (which
enforces the named ``date`` group), ``extract_filename_date``, and
``normalize_authored_date`` — extraction and date-validity are identical to
what the actual backfill would do, so a high ``match_rate`` here predicts a
clean backfill.

Fields:
    --regex REGEX        Required. Must contain a named group 'date', e.g.
                         '(?P<date>\d{4}-\d{2}-\d{2})'. A malformed regex or a
                         missing 'date' group is a usage error (INVALID_REGEX).
    --field {filename,body}
                         filename (default): match each document's file_name.
                         Cheap — probes the full corpus unless --sample caps it.
                         body: match a sample of document body text. ALWAYS
                         sampled (sampled:true), since a full-corpus body scan
                         is expensive.
    --sample N           Cap the population probed to N documents (default 200).
                         For --field filename it bounds an otherwise-full scan;
                         for --field body it is the (always-on) sample size.
    --file-like PATTERN  Scope to documents whose file_name matches this SQL
                         LIKE pattern (repeatable, OR'd), consistent with
                         scan / search / list_documents.

Output (JSON):
    {
      "field": "filename"|"body",
      "regex": str,
      "sampled": bool,             # true if the probe saw a capped sample
      "sample_size": int|null,     # the --sample cap when sampling, else null
      "population": int,           # documents actually probed
      "matched": int,              # documents whose file_name/body the regex hit
      "match_rate": float,         # matched / population (0.0 when population==0)
      "normalized_ok": int,        # matched AND a valid YYYY-MM-DD calendar date
      "normalized_invalid": int,   # matched but NOT a valid YYYY-MM-DD
      "examples": [                # a few good hits, for eyeballing
        {"file_name": str, "extracted": str, "normalized": str}, ...
      ],
      "unmatched_examples": [str], # a few file_names the regex missed
      "suggested_command": str|null,  # the exact backfill line to hand a human,
                                       # emitted ONLY when match_rate is high
      "match_threshold": float,    # the match_rate cutoff for suggested_command
      "filters": {"file_like": [...]}  # present only when --file-like is active
    }

``suggested_command`` is emitted only when ``match_rate`` >= ``MATCH_THRESHOLD``
(0.9): a regex that matches most of the corpus is worth handing to a human; a
sparse one isn't, and prompting with it would waste a round-trip. When no regex
clears the bar the agent should report dates aren't recoverable and proceed
without temporal filtering rather than send the human on a useless fix.
``normalized_invalid`` counts matches whose captured text isn't a real calendar
date (e.g. ``2024-13-40``) — these are NOT good and never reach
``normalized_ok``.
"""

from __future__ import annotations

import argparse
import shlex

from bartleby.ingest.summarize import (
    FilenameDateError,
    compile_filename_date_regex,
    extract_filename_date,
    normalize_authored_date,
)
from bartleby.project import get_active_project
from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import add_file_like_arg, positive_int


# Emit a suggested backfill command only when the regex clears this match rate.
# 0.9 is deliberately strict: handing a human a command that only dates a
# fraction of the corpus invites a wasted round-trip. Documented in the
# docstring and the GH-0538 decision file.
MATCH_THRESHOLD = 0.9

# A few examples each way is enough to eyeball; more just bloats the envelope.
_EXAMPLE_LIMIT = 5


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("probe_dates", __doc__)
    p.add_argument(
        "--regex", required=True,
        help=r"Regex with a named group 'date' (e.g. '(?P<date>\d{4}-\d{2}-\d{2})').",
    )
    p.add_argument(
        "--field", choices=["filename", "body"], default="filename",
        help="filename (default) or body. body is always sampled.",
    )
    p.add_argument(
        "--sample", type=positive_int, default=200,
        help="Cap the population probed (default 200). Always on for --field body.",
    )
    add_file_like_arg(p)
    return p.parse_args(argv)


def _probe_targets(conn, args) -> list[tuple[int, str, str]]:
    """Return ``(document_id, file_name, probe_text)`` rows to match against.

    For ``--field filename`` the probe text is the file_name; for ``--field
    body`` it is the document's concatenated body-chunk text. ``--file-like``
    scopes the document set (OR'd patterns), consistent with scan/search.

    ``--field body`` is always sampled to ``--sample`` (a full-corpus body scan
    is expensive); ``--field filename`` is capped only when ``--sample`` is
    below the matching-document count.
    """
    cur = conn.cursor()
    where = ""
    params: list = []
    if args.file_like:
        clause = " OR ".join("file_name LIKE ?" for _ in args.file_like)
        where = f"WHERE {clause} "
        params = list(args.file_like)

    if args.field == "filename":
        rows = cur.execute(
            f"SELECT document_id, file_name FROM documents {where}"
            "ORDER BY document_id LIMIT ?",
            [*params, args.sample],
        )
        return [(doc_id, name, name) for doc_id, name in rows]

    # body: join document-track chunks, concatenate per document, always sampled.
    rows = cur.execute(
        "SELECT d.document_id, d.file_name, "
        "       GROUP_CONCAT(c.text, ' ') AS body "
        "FROM documents d "
        "JOIN chunks c ON c.source_kind = 'document' AND c.source_id = d.document_id "
        f"{where}"
        "GROUP BY d.document_id, d.file_name "
        "ORDER BY d.document_id LIMIT ?",
        [*params, args.sample],
    )
    return [(doc_id, name, body or "") for doc_id, name, body in rows]


def work(*, conn, args, session_id) -> dict:
    try:
        compiled = compile_filename_date_regex(args.regex)
    except FilenameDateError as e:
        raise SkillError("INVALID_REGEX", str(e)) from e

    targets = _probe_targets(conn, args)
    population = len(targets)

    # --field body is always a sample; --field filename is a sample only when the
    # --sample cap actually bit (population reached the cap, so more may exist).
    sampled = args.field == "body" or population >= args.sample

    matched = 0
    normalized_ok = 0
    normalized_invalid = 0
    examples: list[dict] = []
    unmatched_examples: list[str] = []

    for _doc_id, file_name, probe_text in targets:
        extracted = extract_filename_date(compiled, probe_text)
        if extracted is None:
            if len(unmatched_examples) < _EXAMPLE_LIMIT:
                unmatched_examples.append(file_name)
            continue
        matched += 1
        normalized = normalize_authored_date(extracted)
        if normalized is None:
            normalized_invalid += 1
        else:
            normalized_ok += 1
            if len(examples) < _EXAMPLE_LIMIT:
                examples.append({
                    "file_name": file_name,
                    "extracted": extracted,
                    "normalized": normalized,
                })

    match_rate = matched / population if population else 0.0

    suggested_command = None
    if match_rate >= MATCH_THRESHOLD:
        project = args.project or get_active_project()
        suggested_command = (
            f"bartleby scribe backfill-dates {project} "
            f"--from-filename {shlex.quote(args.regex)}"
        )

    result = {
        "field": args.field,
        "regex": args.regex,
        "sampled": sampled,
        "sample_size": args.sample if sampled else None,
        "population": population,
        "matched": matched,
        "match_rate": round(match_rate, 4),
        "normalized_ok": normalized_ok,
        "normalized_invalid": normalized_invalid,
        "examples": examples,
        "unmatched_examples": unmatched_examples,
        "suggested_command": suggested_command,
        "match_threshold": MATCH_THRESHOLD,
    }
    if args.file_like:
        result["filters"] = {"file_like": args.file_like}
    return result


def main(argv: list[str] | None = None) -> None:
    run(tool_name="probe_dates", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
