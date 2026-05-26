"""Additive-only schema upgrades.

The default position is "no backwards compat" — bumps mean re-ingest. This
chain is the one allowed relaxation: when a bump is *purely additive* (new
tables, indexes, nullable columns), a function here lets existing users run
``bartleby project upgrade <name>`` instead. Non-additive bumps simply have
no entry; ``project upgrade`` refuses.

The codebase never branches on schema version — ``SCHEMA_VERSION`` stays
pinned. This chain is the one-shot gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import apsw

from bartleby.db.schema import SCHEMA_VERSION


def _upgrade_v4_to_v5(conn: apsw.Connection) -> None:
    conn.cursor().execute(
        "ALTER TABLE summaries ADD COLUMN authored_date TEXT"
    )


_UPGRADES: dict[int, Callable[[apsw.Connection], None]] = {
    4: _upgrade_v4_to_v5,
}


def upgrade(conn: apsw.Connection, current_version: int) -> None:
    """Walk the chain from ``current_version`` up to ``SCHEMA_VERSION``.

    Raises if a step is missing — non-additive bumps have no entry and force
    re-ingest.
    """
    v = current_version
    while v < SCHEMA_VERSION:
        step = _UPGRADES.get(v)
        if step is None:
            raise RuntimeError(
                f"No additive upgrade from v{v} to v{v + 1}. "
                f"This is a non-additive bump; re-ingest is required."
            )
        with conn:
            step(conn)
        v += 1

    with conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION),),
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('upgraded_at', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
