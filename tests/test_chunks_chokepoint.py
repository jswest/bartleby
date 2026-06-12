"""Static guard for the polymorphic-chunks chokepoint invariant.

All writes to the polymorphic ``chunks`` table (and its parallel ``chunks_fts``
/ ``chunks_vec`` virtual tables) must flow through the typed helpers in
``bartleby/db/chunks.py`` -- see that module's docstring and ARCHITECTURE.md.
A raw ``INSERT INTO chunks`` anywhere else would let the three tables drift out
of sync without code review noticing.

This test walks every ``bartleby/**/*.py`` source file and fails if any module
other than the allowlisted helper contains an ``INSERT INTO`` targeting one of
those tables. It scans raw text on purpose: the helper writes its SQL as string
literals, so we cannot strip strings before matching.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "bartleby"

# The one module sanctioned to issue raw INSERTs against the chunks tables.
ALLOWLIST = {PACKAGE_ROOT / "db" / "chunks.py"}

# Match ``INSERT INTO <table>`` where <table> is exactly one of the chunks
# tables. ``\s+`` between INTO and the table tolerates the newline-split SQL in
# chunks.py. The ``(?![\w.])`` after the table name is a precise right boundary:
# it rejects longer identifiers (e.g. ``chunks_archive``) so ``chunks`` does not
# falsely match a prefix of ``chunks_fts``/``chunks_vec`` or any other table,
# while still allowing ``chunks(`` / ``chunks `` / end-of-string.
#
# The ``(?!\s*\(\s*chunks_fts\s*,\s*rank\b)`` exclusion lets through *only* the
# single load-bearing FTS5 command form Bartleby emits — the integrity-check
# probe ``INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)``
# in ``bartleby/integrity.py``. Naming the table as the first column with a
# ``rank`` arg is the FTS5 command convention for that probe, a vacuous write
# that inserts no chunk row. Every other command form (notably the destructive
# ``chunks_fts(chunks_fts) VALUES('delete-all')``) and every real external-content
# row write (``rowid``/``text`` columns) is still caught.
CHUNKS_INSERT = re.compile(
    r"INSERT\s+INTO\s+(?:chunks_fts|chunks_vec|chunks)(?![\w.])"
    r"(?!\s*\(\s*chunks_fts\s*,\s*rank\b)",
    re.IGNORECASE,
)


def _python_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_chunks_inserts_only_in_helper() -> None:
    violations: list[str] = []
    for path in _python_sources():
        if path in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for match in CHUNKS_INSERT.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            rel = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(f"{rel}:{line}: {match.group(0)!r}")

    assert not violations, (
        "Raw INSERT INTO chunks/chunks_fts/chunks_vec found outside "
        "bartleby/db/chunks.py. All chunk writes must go through the typed "
        "helpers in that module:\n  " + "\n  ".join(violations)
    )


def test_guard_detects_a_planted_violation() -> None:
    """The matcher fires on a real INSERT and ignores near-misses."""
    assert CHUNKS_INSERT.search("INSERT INTO chunks (source_kind) VALUES (?)")
    assert CHUNKS_INSERT.search("insert into chunks_fts(rowid, text) VALUES (?, ?)")
    assert CHUNKS_INSERT.search("INSERT  INTO\n    chunks_vec(rowid) VALUES (?)")

    # Near-misses that must NOT trip the guard.
    assert not CHUNKS_INSERT.search("SELECT * FROM chunks")
    assert not CHUNKS_INSERT.search("DELETE FROM chunks_fts WHERE rowid = ?")
    assert not CHUNKS_INSERT.search("INSERT INTO chunks_archive VALUES (?)")
    assert not CHUNKS_INSERT.search("# we never INSERT INTO chunkside tables")
    assert not CHUNKS_INSERT.search("INSERT INTO documents VALUES (?)")
    # Only the integrity-check command form (table name first, ``rank`` arg) — a
    # vacuous probe, not a chunk write — is exempt.
    assert not CHUNKS_INSERT.search(
        "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)"
    )
    # A real external-content row write into the same table is still caught.
    assert CHUNKS_INSERT.search(
        "INSERT INTO chunks_fts(rowid, text, section_heading) VALUES (?, ?, ?)"
    )
    # Negative control: the destructive ``delete-all`` command form names the
    # table as its first column too, but lacks the ``rank`` arg, so the narrowed
    # exemption must NOT admit it — it still trips the guard.
    assert CHUNKS_INSERT.search(
        "INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')"
    )


def test_helper_module_is_present_and_does_insert() -> None:
    """Sanity: the allowlisted helper exists and actually owns the INSERTs."""
    (helper,) = ALLOWLIST
    assert helper.exists(), f"missing chunks helper at {helper}"
    assert CHUNKS_INSERT.search(helper.read_text(encoding="utf-8")), (
        "chunks.py no longer contains an INSERT INTO chunks -- the chokepoint "
        "may have moved; update this guard."
    )
