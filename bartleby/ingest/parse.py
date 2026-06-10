"""The per-document unit carried across the scribe ingest phases.

Pulled out of ``commands/scribe.py`` (#306). ``parse_all`` — the phase-1 drain
that produces these — lands here in the next commit; for now this module owns
the dataclass the caption and summarize phases consume.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocUnit:
    """A document parsed + persisted this run (or resumed from an earlier one),
    carried through the caption and summarize phases. ``stages`` accumulates
    per-stage seconds only under ``--timings`` — it's None on the fast path."""
    document_id: int
    file_name: str
    file_hash: str
    page_count: int | None = None
    stages: dict[str, float] | None = None
