#!/usr/bin/env python3
"""assign_tag — attach one tag to one or more documents, bypassing the classifier.

The companion to ``tag``, which decides membership with an LLM that reads the
document *summary*. Reach for ``assign_tag`` when membership was determined
out-of-band — a body scan, a deterministic rule, a human call — especially for
body-level properties (OCR quality, language, "contains tables") that leave no
trace in the summary the classifier sees.

One tag, any number of documents in a single process start (``--documents
1,2,3``) — so a corpus-wide tagging pass pays interpreter startup once, not
once per document. Idempotent: re-assigning an existing pair is a no-op (the
underlying insert is ``OR IGNORE``). An id with no document is reported in
``not_found`` and skipped; the rest of the batch still applies.

**Manual value path.** For a value-tag (one created with --value-type/--pattern)
whose regex can't reach the value — odd formatting, or a value read by eye —
pass ``--value`` to record it manually, mirroring manual tag assignment. The
value is cast/normalized per the tag's value_type exactly as ``extract`` would,
then written to every named document (one value per (tag, document)). Optionally
anchor it to a chunk with ``--chunk <id>`` (the citation source). ``--value`` is
rejected on a plain boolean tag; ``--chunk`` requires ``--value``.

Output:
    {"tag_id": int, "tag": str,
     "value": str|null,            # the cast value, when --value was given
     "chunk_id": int|null,
     "assigned": [{"document_id": int, "file_name": str}, ...],
     "not_found": [int, ...]}
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts._common import comma_int_list, positive_int
from bartleby.skill_scripts._tags import (
    assign,
    cast_value,
    require_tag_by_name,
    require_value_tag,
    resolve_documents,
    upsert_value,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("assign_tag", __doc__)
    p.add_argument(
        "--documents", type=comma_int_list("document_id"), required=True,
        dest="document_ids", help="Comma-separated document ids, e.g. 1,2,3.",
    )
    p.add_argument("--tag", type=str, required=True)
    p.add_argument(
        "--value", type=str, default=None,
        help="Manually record this value for a value-tag (cast per its "
             "value_type). Only valid on a value-tag.",
    )
    p.add_argument(
        "--chunk", type=positive_int, default=None, dest="chunk_id",
        help="Chunk to anchor a manual --value to (its citation source).",
    )
    return p.parse_args(argv)


def work(*, conn, args, session_id) -> dict:
    tag = require_tag_by_name(conn, args.tag)

    cast = None
    if args.value is not None:
        require_value_tag(tag)
        cast = cast_value(tag.value_type, args.value)
        if args.chunk_id is not None and not conn.cursor().execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ?", (args.chunk_id,),
        ).fetchone():
            raise SkillError(
                "UNKNOWN_CHUNK", f"No chunk with chunk_id {args.chunk_id}.",
            )
    elif args.chunk_id is not None:
        raise SkillError(
            "CHUNK_WITHOUT_VALUE", "--chunk requires --value.",
        )

    found, not_found = resolve_documents(conn, args.document_ids)
    for document_id, _ in found:
        if cast is not None:
            upsert_value(conn, document_id, tag.tag_id, cast, args.chunk_id)
        else:
            assign(conn, document_id, [tag.tag_id])

    return {
        "tag_id": tag.tag_id,
        "tag": tag.name,
        "value": cast,
        "chunk_id": args.chunk_id if cast is not None else None,
        "assigned": [{"document_id": d, "file_name": f} for d, f in found],
        "not_found": not_found,
    }


def main(argv: list[str] | None = None) -> None:
    run(
        tool_name="assign_tag", parse_args=parse_args, work=work, argv=argv,
        mutates=True,
    )


if __name__ == "__main__":
    main()
