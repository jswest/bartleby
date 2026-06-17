"""Tests for the chunk-citation backfill helper (scripts/backfill_chunk_citations.py).

The script lives under scripts/ (not an installed package), so it's loaded by
path. Only the pure rewrite is exercised — the regex must touch bare ``[^N]``
and nothing else, and must be idempotent.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "backfill_chunk_citations.py"
_spec = importlib.util.spec_from_file_location("backfill_chunk_citations", _PATH)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


def _make_db(path, bodies):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE findings (finding_id INTEGER PRIMARY KEY, body TEXT)")
    con.executemany("INSERT INTO findings (finding_id, body) VALUES (?, ?)", list(enumerate(bodies)))
    con.commit()
    con.close()


def _bodies(path):
    con = sqlite3.connect(path)
    rows = [b for b, in con.execute("SELECT body FROM findings ORDER BY finding_id")]
    con.close()
    return rows


def test_rewrites_bare_markers():
    body = "Claim one[^593] and two[^228752]."
    new, count = backfill.rewrite_body(body)
    assert new == "Claim one[^chunk:593] and two[^chunk:228752]."
    assert count == 2


def test_leaves_typed_and_external_markers_untouched():
    body = "Chunk[^chunk:12] url[^url:https://x.test/a] doc[^doc:ptr-1] all stay."
    new, count = backfill.rewrite_body(body)
    assert new == body
    assert count == 0


def test_idempotent():
    body = "First[^593] pass."
    once, _ = backfill.rewrite_body(body)
    twice, count = backfill.rewrite_body(once)
    assert twice == once
    assert count == 0


def test_mixed_body_rewrites_only_bare():
    body = "Bare[^593], typed[^chunk:7], external[^url:https://x.test]."
    new, count = backfill.rewrite_body(body)
    assert new == "Bare[^chunk:593], typed[^chunk:7], external[^url:https://x.test]."
    assert count == 1


def test_no_markers_no_change():
    body = "Plain prose with no citations at all."
    new, count = backfill.rewrite_body(body)
    assert new == body
    assert count == 0


def test_backfill_db_writes_and_handles_null_body(tmp_path):
    db = tmp_path / "bartleby.db"
    _make_db(db, ["Bare[^593] and[^chunk:7].", None, "No markers."])
    changed, markers = backfill._backfill_db(db, write=True, backup=True)
    assert (changed, markers) == (1, 1)  # NULL and marker-free rows untouched
    assert _bodies(db) == ["Bare[^chunk:593] and[^chunk:7].", None, "No markers."]


def test_backfill_db_dry_run_does_not_write(tmp_path):
    db = tmp_path / "bartleby.db"
    _make_db(db, ["Bare[^593]."])
    changed, markers = backfill._backfill_db(db, write=False, backup=True)
    assert (changed, markers) == (1, 1)
    assert _bodies(db) == ["Bare[^593]."]  # unchanged
    assert not (tmp_path / ("bartleby.db" + backfill._BACKUP_SUFFIX)).exists()


def test_backfill_db_backup_is_pristine_and_not_clobbered(tmp_path):
    db = tmp_path / "bartleby.db"
    bak = tmp_path / ("bartleby.db" + backfill._BACKUP_SUFFIX)
    _make_db(db, ["Bare[^593]."])
    backfill._backfill_db(db, write=True, backup=True)
    assert _bodies(bak) == ["Bare[^593]."]  # backup holds pre-write state

    # A second --write run is idempotent (0 edits) and leaves the pristine backup.
    changed, markers = backfill._backfill_db(db, write=True, backup=True)
    assert (changed, markers) == (0, 0)
    assert _bodies(bak) == ["Bare[^593]."]
    assert _bodies(db) == ["Bare[^chunk:593]."]
