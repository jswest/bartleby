#!/usr/bin/env python3
"""tag — classify documents against the controlled vocabulary.

Two modes, switched by ``--tag <name>``:
  - Full-vocabulary (no ``--tag``): one LLM call per document picks from the
    entire vocabulary. Best for initial sweeps.
  - Single-tag (``--tag <name>``): one LLM call per document answers "does
    this single tag apply?" Best for adding a new tag to an established
    corpus.

Scope:
  - ``--document-id <id>``: one document.
  - ``--all``: every document with a summary.
  - ``--force``: re-classify even when relevant assignments already exist.

``--force`` correction is asymmetric between the two modes:
  - Single-tag: corrective *both ways*. The fresh verdict assigns the tag when
    it applies and unassigns it when it does not, so a forced re-sweep can both
    add and *remove* that tag.
  - Full-vocab: *additive-only*. A forced re-sweep re-applies and extends the
    classifier's set, but it never unassigns tags outside that set. It cannot
    correct prior over-tagging — stale tags survive a full-vocab ``--force``.
    To remove an over-applied tag, re-run in single-tag mode (``--tag <name>
    --force``), which will unassign it where it no longer applies.

Default skipping (no ``--force``):
  - Full-vocab: skip documents with *any* existing tag assignment.
  - Single-tag: skip documents already assigned that specific tag.

Documents without a summary are reported as ``no_summary`` and skipped —
classification reads the summary, not the body.

Output:
    {
      "mode": "full-vocab" | "single-tag",
      "tag": str|null,
      "model": str,
      "classified": [
        {"document_id": "document:<id>", "file_name": str,
         "assigned_tag_ids": ["tag:<id>", ...],   # full-vocab mode
         "applies": bool                          # single-tag mode
        }
      ],
      "skipped": [
        {"document_id": "document:<id>", "file_name": str, "reason": "already_tagged"|"no_summary"}
      ],
      "failed": [
        {"document_id": "document:<id>", "file_name": str, "error": str}
      ]
    }

A classifier error on a single document never aborts the run: the document is
recorded in ``failed`` and the loop moves on to the next one. Each
classification is retried once before being recorded as failed, since provider
errors (notably an empty Ollama response) are often transient.
"""

from __future__ import annotations

import argparse

from bartleby.skill_runner import SkillError, build_arg_parser, run
from bartleby.skill_scripts import _tags as tags_helpers
from bartleby.skill_scripts._ids import format_output_ids, prefixed_int
from bartleby.skill_scripts._tags import (
    assign,
    classify_full_vocabulary,
    classify_single_tag,
    fetch_vocabulary,
    require_tag_by_name,
    summary_for,
    unassign,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = build_arg_parser("tag", __doc__)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--document-id", type=prefixed_int("document"), default=None,
        dest="document_id", help="Type-tagged document id, e.g. document:204.",
    )
    scope.add_argument("--all", action="store_true", dest="all_documents")
    p.add_argument(
        "--tag", type=str, default=None,
        help="If set, classify against only this single tag.",
    )
    p.add_argument(
        "--force", action="store_true",
        help=(
            "Re-classify even when relevant assignments already exist. "
            "Corrective both ways in single-tag mode (can add or remove the "
            "tag); additive-only in full-vocab mode (re-applies/extends tags "
            "but never removes prior over-tags)."
        ),
    )
    return p.parse_args(argv)


def _target_documents(conn, args) -> list[tuple[int, str]]:
    cur = conn.cursor()
    if args.document_id is not None:
        row = cur.execute(
            "SELECT document_id, file_name FROM documents WHERE document_id = ?",
            (args.document_id,),
        ).fetchone()
        if row is None:
            raise SkillError(
                "DOCUMENT_NOT_FOUND",
                f"No document with id {args.document_id}.",
            )
        return [row]
    return cur.execute(
        "SELECT document_id, file_name FROM documents ORDER BY document_id"
    ).fetchall()


def _already_tagged(conn, document_id: int, tag_id: int | None) -> bool:
    """True if the document carries ``tag_id`` (single-tag mode) or any
    tag (full-vocab mode). Drives the default-skipping policy."""
    cur = conn.cursor()
    if tag_id is None:
        return cur.execute(
            "SELECT 1 FROM document_tags WHERE document_id = ? LIMIT 1",
            (document_id,),
        ).fetchone() is not None
    return cur.execute(
        "SELECT 1 FROM document_tags WHERE document_id = ? AND tag_id = ?",
        (document_id, tag_id),
    ).fetchone() is not None


# One retry per document: provider errors (e.g. an empty Ollama response) are
# often transient, and a single bad call shouldn't drop a document from a sweep.
_CLASSIFY_ATTEMPTS = 2


def _classify_document(
    conn, *, provider, model, temperature,
    document_id, summary, single_tag, vocabulary, force,
) -> dict:
    """Classify one document, write its assignments, and return the verdict
    payload (the caller attaches ``document_id``/``file_name``). Raises on
    provider/classifier failure — the caller decides what to do with that
    (retry, then record in ``failed``)."""
    if single_tag is not None:
        applies = classify_single_tag(
            provider, model, temperature,
            summary_text=summary, tag=single_tag,
        )
        if applies:
            assign(conn, document_id, [single_tag.tag_id])
        elif force:
            unassign(conn, document_id, single_tag.tag_id)
        return {"applies": applies}
    tag_ids = classify_full_vocabulary(
        provider, model, temperature,
        summary_text=summary, vocabulary=vocabulary,
    )
    assign(conn, document_id, tag_ids)
    return {"assigned_tag_ids": tag_ids}


def work(*, conn, args, session_id) -> dict:
    provider, model, temperature = tags_helpers.resolve_classifier()

    single_tag = None
    if args.tag is not None:
        single_tag = require_tag_by_name(conn, args.tag)

    vocabulary = fetch_vocabulary(conn)
    if not vocabulary:
        raise SkillError(
            "EMPTY_VOCABULARY",
            "No tags defined yet. Create one with `add_tag` first.",
        )

    classified: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    for document_id, file_name in _target_documents(conn, args):
        summary = summary_for(conn, document_id)
        if summary is None:
            skipped.append({
                "document_id": document_id, "file_name": file_name,
                "reason": "no_summary",
            })
            continue

        if not args.force and _already_tagged(
            conn, document_id, single_tag.tag_id if single_tag else None,
        ):
            skipped.append({
                "document_id": document_id, "file_name": file_name,
                "reason": "already_tagged",
            })
            continue

        ident = {"document_id": document_id, "file_name": file_name}
        for _ in range(_CLASSIFY_ATTEMPTS):
            try:
                verdict = _classify_document(
                    conn, provider=provider, model=model, temperature=temperature,
                    document_id=document_id, summary=summary,
                    single_tag=single_tag, vocabulary=vocabulary, force=args.force,
                )
                classified.append({**ident, **verdict})
                break
            except Exception as e:  # noqa: BLE001 — one bad doc must not abort the sweep
                error = f"{type(e).__name__}: {e}"
        else:
            failed.append({**ident, "error": error})

    return format_output_ids({
        "mode": "single-tag" if single_tag is not None else "full-vocab",
        "tag": single_tag.name if single_tag is not None else None,
        "model": model,
        "classified": classified,
        "skipped": skipped,
        "failed": failed,
    })


def main(argv: list[str] | None = None) -> None:
    # NOT mutates=True — deliberately unwrapped (issue #340). Unlike the other
    # write scripts, work() loops one LLM classification per document, possibly
    # `--all` over the whole corpus, against the user's often-busy local Ollama.
    # Per-document failure is tolerated by design (the `failed` bucket, ~:182-195)
    # and resume is cheap via the `_already_tagged` skip (~:100-112). Wrapping the
    # whole sweep in one transaction would hold a write lock for minutes-to-hours
    # AND roll back every already-classified document on a mid-sweep failure,
    # destroying that resumability. Incremental per-document commit is correct.
    run(tool_name="tag", parse_args=parse_args, work=work, argv=argv)


if __name__ == "__main__":
    main()
