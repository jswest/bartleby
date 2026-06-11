"""Helpers shared across skill scripts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_finding_chunks,
)
from bartleby.skill_runner import SkillError


# Inline citation marker: standard markdown footnote syntax with a chunk_id.
_CITATION_MARKER = re.compile(r"\[\^(\d+)\]")
# Bare ``[N]`` — looks like a citation, isn't one. Doesn't match ``[^N]`` because
# the ``^`` sits between the bracket and the digits.
_MALFORMED_MARKER = re.compile(r"\[(\d+)\]")


def extract_citations(body: str) -> list[int]:
    """Return chunk_ids from ``[^N]`` markers in first-appearance order, deduped."""
    seen: dict[int, None] = {}
    for m in _CITATION_MARKER.finditer(body):
        seen[int(m.group(1))] = None
    return list(seen)


def reject_malformed_citations(body: str) -> None:
    """Raise ``MALFORMED_CITATION`` if the body contains ``[N]`` (missing ``^``).

    These markers render as bracketed prose but are silently ignored by the
    citation extractor, so the claim ends up effectively uncited. Refuse
    loudly so the agent fixes the typo before persisting.
    """
    bad = [m.group(0) for m in _MALFORMED_MARKER.finditer(body)]
    if not bad:
        return
    deduped = list(dict.fromkeys(bad))
    raise SkillError(
        "MALFORMED_CITATION",
        f"Found {len(bad)} citation-shaped markers missing the caret: "
        f"{', '.join(deduped)}. Write citations as [^N], not [N].",
        malformed_markers=deduped,
    )


def memory_enabled(conn, session_id: int) -> bool:
    """Whether ``session_id`` has memory enabled.

    Findings are the agent's persistent memory. When memory is off, the read
    paths narrow what's visible so an evaluation run isn't contaminated by
    other sessions' conclusions:

    - ``search`` silently drops *all* findings from a mixed result list (the
      load-bearing invariant — ranked retrieval is the contamination vector).
    - the direct-read commands (``list_findings`` / ``read_finding``) restrict
      to the session's *own* findings: a run can always read back what it just
      wrote, but never another session's work.

    The runner resolves/creates the active session before work() runs, so the
    row is guaranteed to exist (same assumption as search.py).
    """
    return bool(conn.cursor().execute(
        "SELECT memory_enabled FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0])


def owned_finding_ids(conn, finding_ids, session_id: int) -> set[int]:
    """Subset of ``finding_ids`` authored by ``session_id``.

    The single ownership query behind the memory wall's finding half. Callers
    pass ``finding_id``s (for finding-kind chunks, that's the chunk's
    ``source_id``). Ids absent from ``findings`` — or owned by another session —
    are simply not in the returned set.
    """
    ids = list(finding_ids)
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    return {
        row[0]
        for row in conn.cursor().execute(
            f"SELECT finding_id FROM findings "
            f"WHERE finding_id IN ({placeholders}) AND session_id = ?",
            (*ids, session_id),
        )
    }


def assert_findings_accessible(
    conn, session_id: int, finding_ids, *, action: str,
) -> None:
    """Enforce the memory wall's finding-ownership half over ``finding_ids``.

    The invariant, expressed once: a ``memory_enabled=0`` session may *touch* a
    finding only if it authored it — reading another session's finding (or its
    chunks) would contaminate an evaluation with prior conclusions. A no-op when
    memory is enabled (every finding is accessible). Otherwise, if any id was
    authored by another session, raise ``MEMORY_OFF`` naming the foreign id(s)
    in ``foreign_finding_ids``; ``action`` (``"read"`` / ``"edit"`` /
    ``"delete"`` / ``"merge"``) fills the message tail.

    Every gating site routes through here so the wall is expressed in one place,
    not re-derived per script. (``search``'s blanket result-filtering and
    ``read_chunks``'s ``--chunks`` drop-to-``missing`` are the wall's *silent*
    halves; they share :func:`owned_finding_ids` but don't raise.)
    """
    if memory_enabled(conn, session_id):
        return
    requested = list(finding_ids)
    foreign = sorted(set(requested) - owned_finding_ids(conn, requested, session_id))
    if not foreign:
        return
    if len(foreign) == 1:
        subject = (
            f"finding {foreign[0]} was authored by another session, so it is "
            "not accessible"
        )
    else:
        subject = (
            f"findings {foreign} were authored by another session, so they are "
            "not accessible"
        )
    raise SkillError(
        "MEMORY_OFF",
        f"This session has memory disabled and {subject}. Start a "
        f"memory-enabled session (omit --no-memory) to {action} other "
        "sessions' findings.",
        foreign_finding_ids=foreign,
    )


def validate_chunk_ids_exist(conn, chunk_ids: list[int]) -> None:
    """Raise ``UNKNOWN_CITATIONS`` if any chunk_id is missing from ``chunks``."""
    if not chunk_ids:
        return
    ph = ",".join("?" * len(chunk_ids))
    cur = conn.cursor()
    seen = {
        row[0] for row in cur.execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({ph})", chunk_ids,
        )
    }
    missing = sorted(set(chunk_ids) - seen)
    if missing:
        raise SkillError(
            "UNKNOWN_CITATIONS",
            f"Inline citations reference unknown chunk_ids: {missing}. "
            "Each [^N] marker must be a real chunk_id in this project.",
            unknown_chunk_ids=missing,
        )


def reject_citations_to_involved_findings(
    conn, citations: list[int], finding_ids, *, code: str, action: str,
) -> None:
    """Reject ``[^N]`` markers citing finding-kind chunks the op will destroy.

    Findings may legitimately cite *finding-kind* chunks, but the merge/edit
    write paths delete-and-rebuild the body chunks of every finding *involved*
    in the operation. If the new body cites one of those soon-to-die chunk ids,
    two silent failures follow (both via ``finding_citations`` →
    ``chunks(chunk_id) ON DELETE CASCADE``): a citation row inserted while the
    chunk still exists gets cascade-deleted, leaving a dangling ``[^N]``; or the
    chunk is deleted *before* the citation is replaced, surfacing as an opaque
    ``INTERNAL_ERROR`` from the FK violation. One upfront ownership check on the
    cited *chunk rows* (their ``source_kind='finding'`` + owning ``source_id``)
    covers both — never by matching body text. Raises ``code`` (e.g.
    ``CITES_MERGED_CHUNKS`` / ``CITES_OWN_CHUNKS``) naming the offending chunk
    ids in ``offending_chunk_ids`` so the agent knows which markers to fix.
    """
    ids = list(finding_ids)
    if not citations or not ids:
        return
    cite_ph = ",".join("?" * len(citations))
    find_ph = ",".join("?" * len(ids))
    offending = sorted(
        row[0] for row in conn.cursor().execute(
            f"SELECT chunk_id FROM chunks "
            f"WHERE source_kind = 'finding' "
            f"AND chunk_id IN ({cite_ph}) AND source_id IN ({find_ph})",
            (*citations, *ids),
        )
    )
    if not offending:
        return
    raise SkillError(
        code,
        f"Inline citations reference chunk_ids {offending}, which belong to "
        f"finding(s) {sorted(ids)} this {action} is about to rewrite — those "
        "chunks will not survive. Remove or repoint those [^N] markers.",
        offending_chunk_ids=offending,
    )


def load_finding_body(conn, body_file: str) -> tuple[str, list[int]]:
    """Read a finding body file and return ``(body, validated_citations)``.

    The single read+validate path shared by ``save_finding``, ``edit_finding``,
    and ``merge_findings``: existence (``BODY_FILE_NOT_FOUND``), non-empty
    (``EMPTY_BODY``), no caret-less ``[N]`` markers (``MALFORMED_CITATION``),
    at least one ``[^N]`` marker (``NO_INLINE_CITATIONS``), and every cited
    chunk_id real (``UNKNOWN_CITATIONS``). Citations are returned in
    first-appearance order, deduped.
    """
    body_path = Path(body_file)
    if not body_path.exists() or not body_path.is_file():
        raise SkillError(
            "BODY_FILE_NOT_FOUND",
            f"--body-file path does not exist: {body_path}",
        )
    body = body_path.read_text(encoding="utf-8")
    if not body.strip():
        raise SkillError("EMPTY_BODY", "Finding body is empty.")

    reject_malformed_citations(body)
    citations = extract_citations(body)
    if not citations:
        raise SkillError(
            "NO_INLINE_CITATIONS",
            "Finding body must include at least one inline citation marker "
            "of the form [^<chunk_id>] (e.g. [^4192]). See SKILL.md.",
        )
    validate_chunk_ids_exist(conn, citations)
    return body, citations


def validated_replacement(
    new_value: str | None, current: str, *, code: str, label: str,
) -> str:
    """Return ``new_value`` (validated non-blank) or ``current`` if unchanged.

    Used by ``edit_finding`` / ``merge_findings`` for optional ``--title`` /
    ``--description`` overrides: omitting the flag keeps the current value;
    passing a blank one is rejected with ``code``.
    """
    if new_value is None:
        return current
    if not new_value.strip():
        raise SkillError(code, f"Finding {label} must be non-empty.")
    return new_value


def embed_body_chunks(body: str) -> list[ChunkInput]:
    """Chunk + embed ``body`` into ``ChunkInput``s, no DB touch (issue #340).

    The EMBED phase of the finding/summary write path, split out from the WRITE
    phase so callers can run it *before* opening any write — embedding doesn't
    need the row's id. ``embed_texts`` is in-process sentence-transformers with a
    ~5-10s lazy model load on first call in a fresh process; under the runner's
    transaction wrap, doing it after the first write would hold the write lock
    across that load (``busy_timeout`` is only 5000ms → ``BusyError`` under
    concurrency). Hoisting it ahead of the first write keeps apsw's deferred
    transaction from grabbing a lock until the millisecond SQL tail. Returns
    ``[]`` for an empty body.

    The chunker + embedder are imported lazily here (not at module top) so the
    FTS-only read scripts that share this module — ``scan``, ``list_documents``,
    ``read_chunks``, ``describe_corpus`` — never pay the embedding/model stack's
    import cost just to import a helper they don't embed with (#371).
    """
    from bartleby.ingest.chunk import chunk_markdown_string
    from bartleby.ingest.embed import embed_texts

    rows = chunk_markdown_string(body)
    if not rows:
        return []
    embeddings = embed_texts([r.text for r in rows])
    return [
        ChunkInput(
            text=row.text,
            embedding=emb,
            chunk_index=i,
            section_heading=row.section_heading,
            content_type=row.content_type,
        )
        for i, (row, emb) in enumerate(zip(rows, embeddings))
    ]


def write_finding_chunks(
    conn, finding_id: int, chunk_inputs: list[ChunkInput]
) -> list[int]:
    """Replace this finding's chunks with pre-embedded ones (WRITE phase).

    Deletes any existing finding chunks and inserts ``chunk_inputs`` (built by
    :func:`embed_body_chunks`) via the typed helper, returning the new chunk_ids
    in insertion order. This is the only part that writes, so callers hoist
    :func:`embed_body_chunks` above their first write and call this at the SQL
    tail. Callers also need to manage ``finding_citations``.
    """
    delete_chunks_for(conn, "finding", finding_id)
    if not chunk_inputs:
        return []
    return insert_finding_chunks(conn, finding_id, chunk_inputs)


def replace_finding_citations(conn, finding_id: int, chunk_ids: list[int]) -> None:
    """Atomically swap ``finding_citations`` rows for this finding."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM finding_citations WHERE finding_id = ?", (finding_id,),
    )
    cur.executemany(
        "INSERT INTO finding_citations (finding_id, chunk_id) VALUES (?, ?)",
        [(finding_id, cid) for cid in chunk_ids],
    )


def session_provenance(conn, session_id: int) -> dict:
    """Return ``{session_name, model, harness}`` for a session.

    The finding-returning scripts all attribute a finding to its authoring
    session and now surface which backend wrote it (issue #62). ``model`` /
    ``harness`` are NULL when the backend was never recorded — never faked.
    """
    row = conn.cursor().execute(
        "SELECT name, model, harness FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    name, model, harness = row
    return {"session_name": name, "model": model, "harness": harness}


def resolve_citations(conn, chunk_ids: list[int]) -> list[dict]:
    """Enrich each cited chunk_id with source_name/file_name/page_number.

    The agent gets this back so it can render human-readable citations in its
    reply alongside the structural chunk_id.
    """
    if not chunk_ids:
        return []
    locations = chunk_locations(conn, chunk_ids)
    names = source_names(
        conn, {(loc["source_kind"], loc["source_id"]) for loc in locations.values()},
    )
    out = []
    for cid in chunk_ids:
        loc = locations.get(cid)
        if loc is None:    # citation chunk vanished between validation and here
            continue
        out.append({
            "chunk_id": cid,
            "source_kind": loc["source_kind"],
            "source_name": names.get((loc["source_kind"], loc["source_id"]), ""),
            "file_name": loc["file_name"],
            "page_number": loc["page_number"],
        })
    return out


def positive_int(value: str) -> int:
    """argparse ``type=`` for an integer >= 1."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def nonneg_int(value: str) -> int:
    """argparse ``type=`` for an integer >= 0."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer") from None
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def add_date_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--authored-after`` / ``--authored-before`` /
    ``--include-nulls`` trio (consumed by ``list_documents``, ``scan``, and
    ``describe_corpus``). The help text is the canonical wording; resolution
    and validation happen in ``_tags.resolve_scope`` / ``validate_date_bound``.
    """
    parser.add_argument(
        "--authored-after",
        type=str, default=None, dest="authored_after",
        help=(
            "Keep documents whose authored_date is on or after this "
            "YYYY-MM-DD. Composable with --tag and --authored-before."
        ),
    )
    parser.add_argument(
        "--authored-before",
        type=str, default=None, dest="authored_before",
        help="Keep documents whose authored_date is on or before this YYYY-MM-DD.",
    )
    parser.add_argument(
        "--include-nulls",
        action="store_true", dest="include_nulls",
        help=(
            "With a date bound active, keep NULL-dated (undated) documents "
            "instead of excluding them. No effect without a date bound."
        ),
    )


def add_file_like_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--file-like`` filename filter (consumed by ``scan``,
    ``search``, and ``list_documents``).

    Repeatable; the patterns OR together and the group ANDs with the other
    scope filters (``--tag`` / ``--in-documents`` / date bounds). Resolution
    (the parameterized ``file_name LIKE`` pushdown) happens in
    ``_tags.resolve_scope``; this is the canonical help wording.
    """
    parser.add_argument(
        "--file-like",
        action="append", default=None, dest="file_like", metavar="PATTERN",
        help=(
            "Restrict to documents whose file_name matches this SQL LIKE "
            "pattern (%% = any run, _ = one char), e.g. 'J000304__%%'. Repeat "
            "for OR; the group ANDs with --tag / --in-documents / date bounds."
        ),
    )


def apply_preview(text: str, preview: int | None) -> str:
    """Truncate ``text`` to ``preview`` chars (appending ``…``); pass through
    unchanged when ``preview`` is ``None`` or the text already fits."""
    if preview is None or len(text) <= preview:
        return text
    return text[:preview] + "…"


def pagination_hint(offset: int, count: int, total: int) -> str | None:
    """The shared ``Showing X-Y of N`` next-page hint, or ``None`` when done.

    ``count`` is the number of rows on the current page. Returns ``None`` when
    the page is empty or already reaches ``total``. Shared by ``list_documents``
    and ``list_findings`` (``scan`` builds a different envelope around its own
    ``total`` and isn't a consumer)."""
    next_offset = offset + count
    if count == 0 or next_offset >= total:
        return None
    return (
        f"Showing {offset + 1}-{next_offset} of {total}. "
        f"Pass --offset {next_offset} to continue."
    )


def finding_chunk_and_citation_ids(
    cur, finding_id: int,
) -> tuple[list[int], list[int]]:
    """Return ``(chunk_ids, citation_ids)`` for a finding.

    ``chunk_ids`` are the finding's own body chunks (``source_kind =
    'finding'``) in ``chunk_index`` order; ``citation_ids`` are the chunks it
    cites (from ``finding_citations``).
    """
    chunk_ids = [
        r[0] for r in cur.execute(
            "SELECT chunk_id FROM chunks WHERE source_kind = 'finding' "
            "AND source_id = ? ORDER BY chunk_index",
            (finding_id,),
        )
    ]
    citation_ids = [
        r[0] for r in cur.execute(
            "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
            (finding_id,),
        )
    ]
    return chunk_ids, citation_ids


def comma_int_list(label: str) -> Callable[[str], list[int]]:
    """argparse ``type=`` factory for a comma-separated list of ints.

    ``label`` names the id being parsed (e.g. ``"document_id"``) so the error
    message is meaningful to the agent reading argparse's failure.
    """
    def _parse(s: str) -> list[int]:
        out: list[int] = []
        for piece in s.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                out.append(int(piece))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"'{piece}' is not an integer {label}"
                ) from None
        if not out:
            raise argparse.ArgumentTypeError(f"at least one {label} required")
        return out
    return _parse


def source_names(
    conn, source_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], str]:
    """Resolve display names for ``(source_kind, source_id)`` pairs in one batch.

    Documents → file_name; summaries → ``"summary of <file_name>"``; findings →
    finding title; images → ``"image in <file_name>, p.<N>"`` (with
    ``" (+K other docs)"`` if the image appears in multiple documents).
    Pairs that don't resolve (deleted underneath us) are simply absent from
    the returned dict.
    """
    by_kind: dict[str, list[int]] = {}
    for kind, sid in source_keys:
        by_kind.setdefault(kind, []).append(sid)

    out: dict[tuple[str, int], str] = {}
    cur = conn.cursor()
    for kind, ids in by_kind.items():
        ph = ",".join("?" * len(ids))
        if kind == "document":
            for did, fname in cur.execute(
                f"SELECT document_id, file_name FROM documents "
                f"WHERE document_id IN ({ph})",
                ids,
            ):
                out[("document", did)] = fname
        elif kind == "summary":
            for sid, fname in cur.execute(
                f"SELECT s.summary_id, d.file_name "
                f"FROM summaries s JOIN documents d USING (document_id) "
                f"WHERE s.summary_id IN ({ph})",
                ids,
            ):
                out[("summary", sid)] = f"summary of {fname}"
        elif kind == "finding":
            for fid, title in cur.execute(
                f"SELECT finding_id, title FROM findings "
                f"WHERE finding_id IN ({ph})",
                ids,
            ):
                out[("finding", fid)] = title
        elif kind == "image":
            out.update(_image_source_names(cur, ids, ph))
    return out


def chunk_locations(
    conn, chunk_ids: list[int]
) -> dict[int, dict]:
    """Resolve {source_kind, source_id, file_name, page_number, authored_date}
    per chunk_id.

    ``file_name`` / ``page_number`` / ``authored_date`` may be ``None`` when not
    applicable. ``authored_date`` is the summarizer-inferred date carried on the
    underlying document's summary row (the same value list_documents and the date
    filters use); it is resolved in the same per-kind join that fetches
    ``file_name``, so it costs no extra query:
      - document chunks: file_name from the document; page_number parsed from
        ``section_heading`` (the 'page N' convention pdfplumber writes);
        authored_date from the document's summary (NULL if undated / no summary).
      - summary chunks: file_name from the underlying document; page_number is
        always None (summaries aren't paginated); authored_date off the summary
        row itself.
      - image chunks: file_name from the primary document the image is linked
        to; page_number from that ``document_images`` join row; authored_date
        from that primary document's summary.
      - finding chunks: file_name + page_number + authored_date all None.

    Chunks whose row was deleted between query and resolution are absent from
    the result entirely.
    """
    if not chunk_ids:
        return {}
    ph = ",".join("?" * len(chunk_ids))
    cur = conn.cursor()
    rows = list(cur.execute(
        f"SELECT chunk_id, source_kind, source_id, page_number "
        f"FROM chunks WHERE chunk_id IN ({ph})",
        chunk_ids,
    ))

    out: dict[int, dict] = {}
    by_kind: dict[str, list[int]] = {}
    for cid, kind, sid, page_number in rows:
        out[cid] = {
            "source_kind": kind, "source_id": sid,
            "file_name": None, "page_number": page_number,
            "authored_date": None,
        }
        by_kind.setdefault(kind, []).append(cid)

    # Resolve file_name + authored_date per source_kind. page_number is already
    # on the row except for image chunks, where the source-of-truth is the
    # document_images join (an image can live on different pages in different
    # documents).
    for kind, cids in by_kind.items():
        sids = list({out[cid]["source_id"] for cid in cids})
        sid_ph = ",".join("?" * len(sids))
        if kind in ("document", "summary"):
            if kind == "document":
                query = (
                    f"SELECT d.document_id, d.file_name, s.authored_date "
                    f"FROM documents d "
                    f"LEFT JOIN summaries s USING (document_id) "
                    f"WHERE d.document_id IN ({sid_ph})"
                )
            else:
                query = (
                    f"SELECT s.summary_id, d.file_name, s.authored_date "
                    f"FROM summaries s JOIN documents d USING (document_id) "
                    f"WHERE s.summary_id IN ({sid_ph})"
                )
            meta = {
                key: (fname, authored_date)
                for key, fname, authored_date in cur.execute(query, sids)
            }
            for cid in cids:
                row = meta.get(out[cid]["source_id"])
                if row:
                    out[cid]["file_name"], out[cid]["authored_date"] = row
        elif kind == "image":
            anchors = _image_anchors(cur, sids)
            for cid in cids:
                anchor = anchors.get(out[cid]["source_id"])
                if anchor:
                    out[cid]["file_name"] = anchor["file_name"]
                    out[cid]["page_number"] = anchor["page_number"]
                    out[cid]["authored_date"] = anchor.get("authored_date")
        # 'finding' falls through with file_name/authored_date=None (and
        # page_number from the column, which is None for finding chunks).
    return out


def _image_anchors(cur, image_ids: list[int]) -> dict[int, dict]:
    """Per-image primary anchor + count of additional documents using it.

    Returns ``{image_id: {file_name, page_number, authored_date,
    other_doc_count}}``. The 'primary' anchor is the lowest
    ``(document_id, page_number)`` join row, matching the existing source_name
    formatting rule; ``authored_date`` is that primary document's
    summarizer-inferred date (NULL if undated / no summary), folded into the
    same join so it costs no extra query.
    """
    if not image_ids:
        return {}
    ph = ",".join("?" * len(image_ids))
    rows = list(cur.execute(
        f"SELECT di.image_id, di.document_id, di.page_number, d.file_name, "
        f"       s.authored_date "
        f"FROM document_images di "
        f"JOIN documents d ON d.document_id = di.document_id "
        f"LEFT JOIN summaries s ON s.document_id = di.document_id "
        f"WHERE di.image_id IN ({ph}) "
        f"ORDER BY di.image_id, di.document_id, di.page_number",
        image_ids,
    ))
    by_image: dict[int, list[tuple[int, int | None, str, str | None]]] = {}
    for image_id, doc_id, page_number, file_name, authored_date in rows:
        by_image.setdefault(image_id, []).append(
            (doc_id, page_number, file_name, authored_date)
        )

    out: dict[int, dict] = {}
    for image_id, occurrences in by_image.items():
        primary_doc, primary_page, primary_name, primary_date = occurrences[0]
        out[image_id] = {
            "file_name": primary_name,
            "page_number": primary_page,
            "authored_date": primary_date,
            "other_doc_count": len({d for d, _, _, _ in occurrences}) - 1,
        }
    return out


def _image_source_names(cur, ids, ph) -> dict[tuple[str, int], str]:
    """Format per-image display names from `_image_anchors`."""
    out: dict[tuple[str, int], str] = {}
    for image_id, anchor in _image_anchors(cur, ids).items():
        page_str = (f", p.{anchor['page_number']}"
                    if anchor["page_number"] is not None else "")
        suffix = (f" (+{anchor['other_doc_count']} other docs)"
                  if anchor["other_doc_count"] > 0 else "")
        out[("image", image_id)] = f"image in {anchor['file_name']}{page_str}{suffix}"
    return out
