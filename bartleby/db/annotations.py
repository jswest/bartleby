"""Typed helpers for the ``finding_annotations`` table (#689).

An annotation is a note layered on a finding: optionally anchored to a verbatim
span of the finding's raw markdown body (a W3C-style text-quote selector), and
optionally pointing at a chunk. The finding body is never modified. This module
is the only place annotation SQL lives in Python.
"""

from __future__ import annotations

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
    finding's current body (else :class:`AnchorNotFound`), and that ``chunk_id``
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
    elif locate_anchor(row[0], anchor_exact, anchor_prefix, anchor_suffix) is None:
        raise AnchorNotFound("anchor does not match the finding body")
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


def list_annotations(conn: apsw.Connection, finding_id: int) -> list[dict]:
    """Annotations on a finding, oldest first, each with ``anchor_found``."""
    rows = conn.cursor().execute(
        f"SELECT {', '.join(_COLUMNS)} FROM finding_annotations "
        "WHERE finding_id = ? ORDER BY annotation_id",
        (finding_id,),
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
    conn: apsw.Connection, finding_ids: list[int]
) -> dict[int, int]:
    """Annotation count per finding id (ids with none map to 0)."""
    counts = {fid: 0 for fid in finding_ids}
    if not finding_ids:
        return counts
    marks = ",".join("?" * len(finding_ids))
    for fid, n in conn.cursor().execute(
        "SELECT finding_id, COUNT(*) FROM finding_annotations "
        f"WHERE finding_id IN ({marks}) GROUP BY finding_id",
        list(finding_ids),
    ):
        counts[fid] = n
    return counts
