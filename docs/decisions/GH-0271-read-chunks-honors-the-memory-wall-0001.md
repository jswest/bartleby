# read_chunks honors the memory wall for foreign finding chunks

**Issue:** #271 · **Status:** settled

## Context

The memory-off wall (ARCHITECTURE.md "Memory-off enforcement") is enforced
script-by-script, not at a chokepoint. `read_finding` gates on ownership
(`read_finding.py:72`: a memory-off session can read its *own* findings but not
another session's), and `search` drops all finding-kind hits wholesale. But
`read_chunks` never referenced memory at all — its `--chunks` and
`--around-chunk` modes return any chunk by id "regardless of source." Because
chunk ids are dense sequential integers and the `--chunks` response's `missing`
list confirms which exist, a memory-off session could enumerate and read the
full text of another session's finding bodies (plus the finding title via
`source_names`) — a verbatim content leak straight through the wall the rest of
the read surface enforces.

## Decision

Extend the same per-script wall into `read_chunks`, scoped to the only chunks
that carry session ownership: `source_kind = 'finding'` (whose `source_id` is
the `findings.finding_id`). Document, summary, and image chunks are unowned and
unaffected.

- **`--chunks`:** when memory is off, finding chunks owned by another session are
  dropped from the result *before* any text or `source_name` is read, so they
  surface in `missing` exactly as a non-existent id would. The leak closes with
  no new "you can't see this" signal beyond what a wrong id already produces.
- **`--around-chunk`:** when memory is off and the target is a foreign finding
  chunk, raise `{"code": "MEMORY_OFF"}`, mirroring `read_finding`. A window never
  crosses `(source_kind, source_id)`, so gating the target also keeps foreign
  finding chunks out of the neighbourhood — no separate neighbour filter needed.
- A session's **own** findings stay fully readable under memory-off (a run can
  verify what it just wrote), and **memory-on** behaviour is unchanged.

## Notes

- **No schema change.** A read-time ownership check; no `SCHEMA_VERSION` bump, no
  re-ingest.
- **The ownership helper stays local to `read_chunks`.** It is *not* promoted to
  `_common` yet, to avoid colliding with the sibling memory-wall issues (#272,
  #275) landing on the same omnibus branch. The broader "give the memory wall a
  single chokepoint, the way `chunks` got one" consolidation is tracked as
  follow-up on omnibus #276, not done here.
- **Smallest fix that fits.** This reproduces `read_finding`'s gate rather than
  introducing a new enforcement mechanism; the `MEMORY_OFF` code and message
  shape match so the agent sees one consistent wall across the read surface.
