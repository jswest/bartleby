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


# Inline corpus-chunk citation marker: ``[^chunk:<chunk_id>]`` (issue #624).
# Type-tagged so the cited id can only ever be a chunk_id — never a document_id or
# finding_id silently mis-stored (#623). ``chunk`` is the canonical *internal*
# scheme of the shared ``[^<scheme>:<ref>]`` grammar (``_EXTERNAL_MARKER`` below):
# it is routed to chunk-citation handling, while ``url`` / ``doc`` stay external.
_CITATION_MARKER = re.compile(r"\[\^chunk:(\d+)\]")
# Citation-shaped but untyped: caret-less ``[N]`` and the now-obsolete bare
# ``[^N]`` chunk form. Both render as bracketed prose but are silently dropped by
# the typed extractor, so both are rejected loudly — see
# :func:`reject_malformed_citations`.
_MALFORMED_MARKER = re.compile(r"\[\^?(\d+)\]")
# Typed citation marker: ``[^<scheme>:<ref>]``. The scheme is an alpha word.
# ``chunk`` is the internal corpus-chunk scheme (matched by ``_CITATION_MARKER``
# above); ``url`` / ``doc`` are external (stored opaque, never fetched);
# ``document`` / ``finding`` are *rejected* — they are not valid citation targets.
_EXTERNAL_MARKER = re.compile(r"\[\^([A-Za-z]+):([^\]]+)\]")
# Schemes an external marker may carry. ``url`` is a web link; ``doc`` is an
# external-dataset document ref (e.g. a filing id). Both are stored opaque.
_EXTERNAL_SCHEMES = ("url", "doc")
# ``chunk`` is internal (handled by ``_CITATION_MARKER``), so the external-marker
# machinery must skip it as it does the rejected schemes — it is neither external
# nor malformed.
_INTERNAL_SCHEME = "chunk"
# Typed schemes that parse as ``[^<scheme>:<ref>]`` but are *not* valid citation
# targets: a document/finding id is never a citation — cite the chunk you were
# handed instead (the #623 confusion, now structurally rejected).
_REJECTED_CITATION_SCHEMES = ("document", "finding")


def text_qualified_fts(fts_expr: str) -> str:
    """Column-qualify an FTS5 MATCH expression to the ``text`` column.

    ``chunks_fts`` indexes both ``text`` and ``section_heading``, so a bare
    ``MATCH (<expr>)`` also fires on heading-only hits — a chunk whose snippet
    never contains the term. Wrapping as ``{text} : (<expr>)`` confines matching
    to the body text, keeping scan and search a strict text-grep. Deliberate
    heading recall stays reachable via scan's ``--heading-like`` and search's
    vector leg. Returns the expression unchanged when empty, so an empty token
    set never becomes a malformed ``{text} : ()``.
    """
    if not fts_expr:
        return fts_expr
    return f"{{text}} : ({fts_expr})"


def extract_citations(body: str) -> list[int]:
    """Return chunk_ids from ``[^chunk:N]`` markers, first-appearance order, deduped."""
    seen: dict[int, None] = {}
    for m in _CITATION_MARKER.finditer(body):
        seen[int(m.group(1))] = None
    return list(seen)


def _well_formed_external(m: re.Match) -> tuple[str, str] | None:
    """``(scheme, ref)`` for a well-formed *external* marker, else ``None``.

    The single well-formedness predicate — a known external scheme (``url`` /
    ``doc``) and a non-blank ref — shared by the skip-on-bad extractor and the
    raise-on-bad save-time check so the rule can't drift between them. Normalizes
    scheme to lowercase and strips the ref. ``chunk`` (internal) and the rejected
    ``document`` / ``finding`` schemes return ``None`` here; they are classified
    separately (the malformed check excludes ``chunk`` and names the rejected
    schemes specifically).
    """
    scheme, ref = m.group(1).lower(), m.group(2).strip()
    if scheme not in _EXTERNAL_SCHEMES or not ref:
        return None
    return scheme, ref


def extract_external_citations(body: str) -> list[dict]:
    """Return external citations from ``[^url:…]`` / ``[^doc:…]`` markers.

    Each entry is ``{"scheme": "url"|"doc", "ref": <opaque str>}`` in
    first-appearance order, deduped on ``(scheme, ref)``. These markers ride
    *alongside* the mandatory ``[^N]`` chunk citations — never as a substitute:
    the scheme is alpha, so :func:`extract_citations` (digit-only) ignores them
    and the ≥1-chunk requirement is unchanged. The ref is opaque — it is not
    parsed, validated as a real URL, or fetched (no network, per the guardrail);
    only its well-formedness (a known scheme, non-blank ref) is checked at save
    time by :func:`reject_malformed_external_citations`. Ill-formed markers that
    somehow reached a stored body are left in the text but not promoted here.
    Computed on read; no DB row backs it (see the body-marker-not-table decision).
    """
    seen: dict[tuple[str, str], None] = {}
    out: list[dict] = []
    for m in _EXTERNAL_MARKER.finditer(body):
        parsed = _well_formed_external(m)
        if parsed is None or parsed in seen:
            continue
        seen[parsed] = None
        out.append({"scheme": parsed[0], "ref": parsed[1]})
    return out


def reject_wrong_typed_citations(body: str) -> None:
    """Raise ``WRONG_CITATION_TYPE`` for ``[^document:N]`` / ``[^finding:N]``.

    A type-tagged marker whose scheme is ``document`` or ``finding`` parses as a
    typed citation but points at the wrong kind of id — the #623 confusion made
    structural: a citation target is *always* a chunk_id you were handed this
    session, never a document_id or finding_id. Refuse loudly so the agent
    repoints the marker to the chunk it actually means (``[^chunk:<id>]``).
    """
    bad = [
        m.group(0) for m in _EXTERNAL_MARKER.finditer(body)
        if m.group(1).lower() in _REJECTED_CITATION_SCHEMES
    ]
    if not bad:
        return
    bad = list(dict.fromkeys(bad))
    raise SkillError(
        "WRONG_CITATION_TYPE",
        f"Found {len(bad)} citation marker(s) targeting a document/finding id: "
        f"{', '.join(bad)}. A citation target is always a chunk_id you were "
        "handed this session — write [^chunk:<chunk_id>], never "
        "[^document:<id>] or [^finding:<id>].",
        wrong_type_markers=bad,
    )


def reject_malformed_internal_citations(body: str) -> None:
    """Raise ``MALFORMED_CITATION`` for a ``[^chunk:<ref>]`` whose ref isn't all-digits.

    A marker like ``[^chunk:42abc]`` carries the internal ``chunk`` scheme, so it
    slips every other guard: it's scrubbed by :func:`reject_malformed_citations`
    (matches the ``[^<scheme>:<ref>]`` grammar), skipped by
    :func:`reject_wrong_typed_citations` (``chunk`` isn't a rejected scheme), and
    skipped by :func:`reject_malformed_external_citations` (``chunk`` is the
    internal scheme, excluded there). But :func:`extract_citations` requires
    ``\\d+``, so a non-digit ref never extracts — the citation is *silently
    dropped*, exactly the loss #624 exists to kill. Refuse loudly so the agent
    fixes the ref before persisting.
    """
    bad = [
        m.group(0) for m in _EXTERNAL_MARKER.finditer(body)
        if m.group(1).lower() == _INTERNAL_SCHEME and not m.group(2).strip().isdigit()
    ]
    if not bad:
        return
    bad = list(dict.fromkeys(bad))
    raise SkillError(
        "MALFORMED_CITATION",
        f"Found {len(bad)} malformed chunk citation marker(s): "
        f"{', '.join(bad)}. A chunk citation must be [^chunk:<int>] "
        "(e.g. [^chunk:4192]) — the ref must be a bare integer chunk_id.",
        malformed_markers=bad,
    )


def reject_malformed_external_citations(body: str) -> None:
    """Raise ``MALFORMED_EXTERNAL_CITATION`` for an ill-formed external marker.

    External markers (``[^<scheme>:<ref>]``) must carry a known scheme
    (``url`` / ``doc``) and a non-blank ref. A marker with an unknown scheme
    (e.g. ``[^ftp:…]``) or an empty ref (``[^url:]``) renders as bracketed prose
    but is silently dropped by :func:`extract_external_citations`, so the
    external attribution is effectively lost. Refuse loudly so the agent fixes
    it before persisting. The internal ``chunk`` scheme (handled as a corpus
    citation) and the rejected ``document`` / ``finding`` schemes (handled by
    :func:`reject_wrong_typed_citations`) are excluded here so they aren't
    double-flagged. The ref itself is never fetched or otherwise validated beyond
    non-blankness — it is stored opaque.
    """
    bad = [
        m.group(0) for m in _EXTERNAL_MARKER.finditer(body)
        if m.group(1).lower() not in (_INTERNAL_SCHEME, *_REJECTED_CITATION_SCHEMES)
        and _well_formed_external(m) is None
    ]
    if not bad:
        return
    bad = list(dict.fromkeys(bad))
    raise SkillError(
        "MALFORMED_EXTERNAL_CITATION",
        f"Found {len(bad)} ill-formed external citation marker(s): "
        f"{', '.join(bad)}. Write them as [^url:<url>] or [^doc:<ref>] "
        f"with a known scheme ({', '.join(_EXTERNAL_SCHEMES)}) and a non-blank "
        "ref.",
        malformed_markers=bad,
    )


def reject_malformed_citations(body: str) -> None:
    """Raise ``MALFORMED_CITATION`` for an untyped citation-shaped marker.

    Both the caret-less ``[N]`` and the now-obsolete bare ``[^N]`` chunk form
    (issue #624) render as bracketed prose but are silently ignored by the typed
    ``[^chunk:N]`` extractor, so the claim ends up effectively uncited. Refuse
    loudly so the agent rewrites them as ``[^chunk:<chunk_id>]`` before persisting.

    Typed markers (``[^<scheme>:<ref>]``) may legitimately carry a ``[N]``-shaped
    substring inside the ref (e.g. a URL ending ``…/doc[3]``), whose closing bracket
    is also the marker's — so scan a copy with typed markers masked out, else a
    valid ``[^chunk:N]`` / ``[^url:…]`` marker false-trips this guard. (Refs may not
    themselves contain ``]``; that terminates the marker — percent-encode it.)
    """
    scrubbed = _EXTERNAL_MARKER.sub(" ", body)
    bad = [m.group(0) for m in _MALFORMED_MARKER.finditer(scrubbed)]
    if not bad:
        return
    deduped = list(dict.fromkeys(bad))
    raise SkillError(
        "MALFORMED_CITATION",
        f"Found {len(bad)} untyped citation-shaped marker(s): "
        f"{', '.join(deduped)}. Write citations as [^chunk:<chunk_id>], "
        "not [N] or [^N].",
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
            "Each [^chunk:<id>] marker must be a real chunk_id in this project.",
            unknown_chunk_ids=missing,
        )


def reject_citations_to_involved_findings(
    conn, citations: list[int], finding_ids, *, code: str, action: str,
) -> None:
    """Reject ``[^chunk:N]`` markers citing finding-kind chunks the op will destroy.

    Findings may legitimately cite *finding-kind* chunks, but the merge/edit
    write paths delete-and-rebuild the body chunks of every finding *involved*
    in the operation. If the new body cites one of those soon-to-die chunk ids,
    two silent failures follow (both via ``finding_citations`` →
    ``chunks(chunk_id) ON DELETE CASCADE``): a citation row inserted while the
    chunk still exists gets cascade-deleted, leaving a dangling ``[^chunk:N]``; or the
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
        "chunks will not survive. Remove or repoint those [^chunk:N] markers.",
        offending_chunk_ids=offending,
    )


def load_finding_body(conn, body_file: str) -> tuple[str, list[int]]:
    """Read a finding body file and return ``(body, validated_citations)``.

    The single read+validate path shared by ``save_finding``, ``edit_finding``,
    and ``merge_findings``: existence (``BODY_FILE_NOT_FOUND``), non-empty
    (``EMPTY_BODY``), no untyped ``[N]`` / ``[^N]`` markers and no non-integer
    ``[^chunk:<ref>]`` ref (``MALFORMED_CITATION``),
    no ``[^document:N]`` / ``[^finding:N]`` wrong-type markers
    (``WRONG_CITATION_TYPE``), every external ``[^url:…]``/``[^doc:…]`` marker
    well-formed (``MALFORMED_EXTERNAL_CITATION``), at least one ``[^chunk:N]`` chunk
    marker (``NO_INLINE_CITATIONS`` — external markers never satisfy this), and
    every cited chunk_id real (``UNKNOWN_CITATIONS``). Citations are returned in
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
    reject_malformed_internal_citations(body)
    reject_wrong_typed_citations(body)
    reject_malformed_external_citations(body)
    citations = extract_citations(body)
    if not citations:
        raise SkillError(
            "NO_INLINE_CITATIONS",
            "Finding body must include at least one inline citation marker "
            "of the form [^chunk:<chunk_id>] (e.g. [^chunk:4192]). See SKILL.md.",
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


def comma_field_list(value: str) -> list[str]:
    """argparse ``type=`` for ``--returning``: a comma-separated field list.

    Splits on commas, trims whitespace, drops empties, and preserves order
    (deduped). Validation against the per-script whitelist is *not* done here —
    argparse can't see the whitelist and a bad field deserves the JSON error
    envelope (``UNKNOWN_RETURNING_FIELD``), not argparse's exit-2 dump. So this
    only rejects a syntactically empty selector; :func:`project_row` does the
    whitelist check at work() time.
    """
    out: list[str] = []
    for piece in value.split(","):
        piece = piece.strip()
        if piece and piece not in out:
            out.append(piece)
    if not out:
        raise argparse.ArgumentTypeError("at least one field required")
    return out


def add_returning_arg(parser: argparse.ArgumentParser, whitelist: list[str]) -> None:
    """Add the shared ``--returning <field>,...`` projection flag.

    ``whitelist`` is the script's selectable field set (in canonical order); it
    appears in the help so ``--help`` names the valid fields. Omitting the flag
    leaves the script's default/brief projection untouched — ``--returning`` is
    purely additive. Resolution + validation happen in :func:`project_row`.
    """
    parser.add_argument(
        "--returning",
        type=comma_field_list, default=None, dest="returning", metavar="FIELD,...",
        help=(
            "Project each row to exactly these comma-separated fields, in the "
            "order given. Overrides the default projection (and --brief). "
            "Selectable fields: " + ", ".join(whitelist) + ". An unknown field "
            "returns an UNKNOWN_RETURNING_FIELD error naming the valid set."
        ),
    )


def validate_returning(requested: list[str] | None, whitelist: list[str]) -> None:
    """Reject an unknown ``--returning`` field up front, independent of row count.

    ``project_row`` validates as it projects, but a query matching zero rows
    never reaches it — so a typo'd field would slip through as a silent empty
    result instead of the ``UNKNOWN_RETURNING_FIELD`` envelope the docstrings and
    SKILL.md promise unconditionally. Call this once at the top of ``work()``
    (before any early empty-result return) against the whitelist that applies to
    the current mode. No-op when ``requested`` is ``None``.
    """
    if requested is None:
        return
    unknown = [f for f in requested if f not in whitelist]
    if unknown:
        raise SkillError(
            "UNKNOWN_RETURNING_FIELD",
            f"--returning got unknown field(s) {unknown}. Valid fields for "
            f"this script: {', '.join(whitelist)}.",
            valid_fields=list(whitelist),
        )


def project_row(
    full: dict, requested: list[str] | None, whitelist: list[str],
) -> dict | None:
    """Project a fully-built row dict down to the ``--returning`` selection.

    ``full`` must carry every field in ``whitelist`` (the caller builds the
    whole row, then this selects). Returns ``None`` when ``requested`` is
    ``None`` — the signal that no ``--returning`` was passed, so the caller
    keeps its own default/brief projection untouched (``--returning`` is purely
    additive). Otherwise returns ``{field: full[field]}`` in the requested
    order. ``chunk_id`` and ``document_id`` are in every script's whitelist by
    construction, so an id is always selectable.

    Raises ``UNKNOWN_RETURNING_FIELD`` (naming the valid set in
    ``valid_fields``) if any requested field isn't whitelisted — the JSON error
    envelope the agent gets back, never argparse's exit-2. ``work()`` should also
    call :func:`validate_returning` up front so this fires even on a zero-row
    query.
    """
    if requested is None:
        return None
    validate_returning(requested, whitelist)
    return {field: full[field] for field in requested}


class CaptureSpec:
    """A compiled ``/regex/`` plus the column names its capture groups project to.

    The shared regex-capture primitive behind ``scan --extract`` and
    ``scan --count-by '/regex/'`` (which is extract-then-group-and-count over the
    same spec). Column naming is fixed once at parse time: a **named** group
    ``(?P<name>...)`` projects to a column ``name``; a **bare** group ``(...)``
    projects to a positional column ``g1``, ``g2``, ... numbered over *all*
    groups (named or not) in pattern order. So ``/(?P<bill>\\d+)-(\\w+)/`` yields
    columns ``["bill", "g2"]``. The numbering follows ``re``'s group indices so a
    positional name never collides with a different group's slot.

    Deliberately **not** a query engine: a spec extracts captured substrings into
    a tidy row of columns; it never filters, joins, or aggregates (that is the
    caller's job — see ``scan``'s #48/#10 boundary).
    """

    __slots__ = ("pattern", "raw", "columns")

    def __init__(self, pattern: re.Pattern, raw: str):
        self.pattern = pattern
        self.raw = raw
        # group index -> column name. Named groups keep their name; bare groups
        # get the positional ``g<N>`` over re's 1-based group indices.
        name_by_index = {idx: name for name, idx in pattern.groupindex.items()}
        self.columns = [
            name_by_index.get(i, f"g{i}") for i in range(1, pattern.groups + 1)
        ]

    def extract_first(self, text: str) -> dict[str, str | None]:
        """Columns from the **first** match in ``text``, or all-``None`` if none.

        The per-chunk ``--extract`` semantics: one row's worth of columns. A
        non-matching pattern yields a cell of ``None`` per column without
        dropping the row, so the caller can union several specs' columns onto one
        chunk row. Within a match, a group that didn't participate
        (``match.group(i)`` is ``None``) is likewise a null cell.
        """
        match = self.pattern.search(text)
        return {
            col: match.group(i) if match else None
            for i, col in enumerate(self.columns, start=1)
        }


def parse_capture_regex(value: str, *, flag: str) -> CaptureSpec:
    """Parse a ``/regex/`` capture pattern into a :class:`CaptureSpec`.

    The single ``/.../``-delimited, compile, require-a-capture-group parse shared
    by ``--extract`` and ``--count-by``. ``flag`` names the originating flag so
    the error envelope is specific. Raises (as the JSON error envelope, never a
    traceback):

    - ``INVALID_CAPTURE_REGEX`` — not ``/.../``-delimited, or doesn't compile.
    - ``CAPTURE_NO_GROUP`` — compiles but carries no capture group (nothing to
      project into a column).
    """
    if not (len(value) >= 2 and value.startswith("/") and value.endswith("/")):
        raise SkillError(
            "INVALID_CAPTURE_REGEX",
            f"{flag} must be a /regex/ delimited by slashes; got {value!r}.",
        )
    pattern_src = value[1:-1]
    try:
        compiled = re.compile(pattern_src)
    except re.error as exc:
        raise SkillError(
            "INVALID_CAPTURE_REGEX",
            f"{flag} regex {value!r} does not compile: {exc}.",
        )
    if compiled.groups < 1:
        raise SkillError(
            "CAPTURE_NO_GROUP",
            f"{flag} regex {value!r} has no capture group; wrap the value to "
            "capture in parentheses, e.g. '/H\\.R\\.\\s*(\\d+)/'.",
        )
    return CaptureSpec(compiled, value)


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
      - document chunks: file_name from the document; page_number read from the
        chunk's first-class ``page_number`` column (the page recorded at ingest);
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
        _, primary_page, primary_name, primary_date = occurrences[0]
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
