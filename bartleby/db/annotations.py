"""Typed helpers for the ``finding_annotations`` table (#689).

An annotation is a note layered on a finding: optionally anchored to a verbatim
span of the finding's raw markdown body (a W3C-style text-quote selector), and
optionally pointing at a chunk. The finding body is never modified. This module
is the only place annotation SQL lives in Python.
"""

from __future__ import annotations

import re

import apsw

_COLUMNS = (
    "annotation_id", "finding_id", "body", "anchor_exact", "anchor_prefix",
    "anchor_suffix", "chunk_id", "is_human_author", "session_id", "created_at",
)


class AnchorNotFound(ValueError):
    """The anchor does not locate in the finding's current body."""


def locate_anchor(
    body: str, exact: str, prefix: str | None = None, suffix: str | None = None
) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the first occurrence of ``exact`` in ``body``.

    When ``prefix``/``suffix`` are given, the occurrence is the first one whose
    preceding text ends with ``prefix`` and/or whose following text starts with
    ``suffix``. ``None`` if nothing matches. Empty ``exact`` raises ValueError.
    """
    if not exact:
        raise ValueError("anchor_exact must be non-empty")
    start = body.find(exact)
    while start != -1:
        end = start + len(exact)
        before_ok = not prefix or body[:start].endswith(prefix)
        after_ok = not suffix or body.startswith(suffix, end)
        if before_ok and after_ok:
            return start, end
        start = body.find(exact, start + 1)
    return None


# Body spans an anchor boundary may not fall strictly inside: a ``[^…]``
# citation marker and a Markdown link's ``](…)`` target. A boundary inside one
# would split it when a reader inserts an inline note marker (``[✎N]``) or a
# highlight at that offset. Each pattern is scanned separately (their matches
# may overlap); the web UI mirrors this rule exactly.
_PROTECTED_SPANS = (re.compile(r"\[\^[^\]]*\]"), re.compile(r"\]\([^)]*\)"))


def splits_protected_span(body: str, start: int, end: int) -> bool:
    """True when ``start`` or ``end`` falls strictly inside a protected span.

    An anchor that wholly contains a marker (or touches one at its edge) is fine.
    """
    return any(
        m.start() < pos < m.end()
        for pattern in _PROTECTED_SPANS
        for m in pattern.finditer(body)
        for pos in (start, end)
    )


def insert_annotation(
    conn: apsw.Connection,
    *,
    finding_id: int,
    body: str,
    is_human_author: bool,
    anchor_exact: str | None = None,
    anchor_prefix: str | None = None,
    anchor_suffix: str | None = None,
    chunk_id: int | None = None,
    session_id: int | None = None,
) -> int:
    """Insert an annotation and return its id.

    Validates that the finding exists, that the anchor (if any) locates in the
    finding's current body without a boundary splitting a citation marker or
    link target (else :class:`AnchorNotFound`), and that ``chunk_id``
    exists when given. Other violations raise ValueError.
    """
    cur = conn.cursor()
    row = cur.execute(
        "SELECT body FROM findings WHERE finding_id = ?", (finding_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"finding {finding_id} does not exist")
    if anchor_exact is None:
        if anchor_prefix is not None or anchor_suffix is not None:
            raise ValueError("anchor_prefix/anchor_suffix require anchor_exact")
    else:
        span = locate_anchor(row[0], anchor_exact, anchor_prefix, anchor_suffix)
        if span is None:
            raise AnchorNotFound(
                "anchor does not occur verbatim in the finding's current body"
            )
        if splits_protected_span(row[0], *span):
            raise AnchorNotFound(
                "anchor boundary falls inside a citation marker or link target"
            )
    if chunk_id is not None and cur.execute(
        "SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone() is None:
        raise ValueError(f"chunk {chunk_id} does not exist")
    cur.execute(
        "INSERT INTO finding_annotations (finding_id, body, anchor_exact, "
        "anchor_prefix, anchor_suffix, chunk_id, is_human_author, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (finding_id, body, anchor_exact, anchor_prefix, anchor_suffix,
         chunk_id, 1 if is_human_author else 0, session_id),
    )
    return conn.last_insert_rowid()


def _with_anchor_found(conn: apsw.Connection, rows: list[tuple]) -> list[dict]:
    """Zip rows into dicts and add ``anchor_found`` against each finding's current body.

    ``anchor_found`` is ``True`` for a whole-finding note (no anchor).
    """
    cur = conn.cursor()
    bodies: dict[int, str] = {}
    out = []
    for r in rows:
        d = dict(zip(_COLUMNS, r))
        fid = d["finding_id"]
        if fid not in bodies:
            row = cur.execute(
                "SELECT body FROM findings WHERE finding_id = ?", (fid,)
            ).fetchone()
            bodies[fid] = row[0] if row else ""
        d["anchor_found"] = d["anchor_exact"] is None or locate_anchor(
            bodies[fid], d["anchor_exact"], d["anchor_prefix"], d["anchor_suffix"]
        ) is not None
        out.append(d)
    return out


def get_annotation(conn: apsw.Connection, annotation_id: int) -> dict | None:
    """One annotation by id (same keys as :func:`list_annotations` rows), or None."""
    rows = conn.cursor().execute(
        f"SELECT {', '.join(_COLUMNS)} FROM finding_annotations "
        "WHERE annotation_id = ?",
        (annotation_id,),
    ).fetchall()
    return _with_anchor_found(conn, rows)[0] if rows else None


# Memory-off wall for notes (#689): a memory-off session sees human notes and
# its own agent notes, never another session's agent notes (those are that
# session's memory). Bound to the walled session id.
_WALL_SQL = " AND (is_human_author = 1 OR session_id = ?)"


def list_annotations(
    conn: apsw.Connection, finding_id: int, *, walled_session_id: int | None = None
) -> list[dict]:
    """Annotations on a finding, oldest first, each with ``anchor_found``.

    ``walled_session_id`` (a memory-off caller) drops other sessions' agent notes.
    """
    params: list = [finding_id]
    wall = ""
    if walled_session_id is not None:
        wall = _WALL_SQL
        params.append(walled_session_id)
    rows = conn.cursor().execute(
        f"SELECT {', '.join(_COLUMNS)} FROM finding_annotations "
        f"WHERE finding_id = ?{wall} ORDER BY annotation_id",
        params,
    ).fetchall()
    return _with_anchor_found(conn, rows)


def delete_annotation(conn: apsw.Connection, annotation_id: int) -> bool:
    """Delete one annotation; True if a row was removed."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM finding_annotations WHERE annotation_id = ?", (annotation_id,)
    )
    return conn.changes() > 0


def count_annotations(
    conn: apsw.Connection,
    finding_ids: list[int],
    *,
    walled_session_id: int | None = None,
) -> dict[int, int]:
    """Annotation count per finding id (ids with none map to 0).

    ``walled_session_id`` applies the same wall as :func:`list_annotations`.
    """
    counts = {fid: 0 for fid in finding_ids}
    if not finding_ids:
        return counts
    marks = ",".join("?" * len(finding_ids))
    params: list = list(finding_ids)
    wall = ""
    if walled_session_id is not None:
        wall = _WALL_SQL
        params.append(walled_session_id)
    for fid, n in conn.cursor().execute(
        "SELECT finding_id, COUNT(*) FROM finding_annotations "
        f"WHERE finding_id IN ({marks}){wall} GROUP BY finding_id",
        params,
    ):
        counts[fid] = n
    return counts


def reparent_annotations(
    conn: apsw.Connection, from_finding_ids: list[int], into_finding_id: int
) -> int:
    """Move every annotation on ``from_finding_ids`` onto ``into_finding_id``.

    Used by a merge so notes on the deleted sources survive (a human correction
    outranks finding text). Their anchors quote the old bodies, so they usually
    read back ``anchor_found: false`` — the note stands, its location is stale.
    Returns the number of annotations moved.
    """
    if not from_finding_ids:
        return 0
    marks = ",".join("?" * len(from_finding_ids))
    conn.cursor().execute(
        f"UPDATE finding_annotations SET finding_id = ? WHERE finding_id IN ({marks})",
        [into_finding_id, *from_finding_ids],
    )
    return conn.changes()
