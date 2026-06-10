# Memory-off finding-ownership check consolidated into one `_common` chokepoint (issue #288)

> Source: [#288](https://github.com/jswest/bartleby/issues/288)

The script-level memory wall (ARCHITECTURE.md "Memory-off enforcement") has two
halves: a *silent* half (`search` drops all findings from results; `read_chunks
--chunks` drops foreign finding chunks into `missing`) and a *raising* half (a
memory-off session touching a foreign finding gets `MEMORY_OFF`). After the #276
hardening sweep landed, the raising half was open-coded at five sites —
`read_finding`, `read_chunks` (`--around-chunk`), `edit_finding`,
`delete_finding`, `merge_findings` — each spelling out `not memory_enabled(conn,
session_id) and owner != session_id` and hand-rolling the `MEMORY_OFF` error.
Five copies of one invariant is the exact drift risk the sweep was hardening
against: a future edit to one site silently diverges, and the wall is only as
strong as its least-careful copy.

This consolidates the check into **one chokepoint** in
`bartleby/skill_scripts/_common.py`, à la the typed `chunks` helpers:

- `owned_finding_ids(conn, finding_ids, session_id) -> set[int]` — the ownership
  query (the authored subset of `finding_ids`). Moved here verbatim from
  `read_chunks._owned_finding_ids`; it's the one place "which findings did this
  session author" is expressed. The silent half (`read_chunks --chunks`) calls
  it directly to filter.
- `assert_findings_accessible(conn, session_id, finding_ids, *, action)` — the
  raising half. A no-op under memory-on; otherwise raises `MEMORY_OFF` naming the
  foreign id(s) in `foreign_finding_ids`, with `action`
  (`read`/`edit`/`delete`/`merge`) filling the message tail. All five raising
  sites route through it; the open-coded checks are deleted.

**No behavior change, no schema change** (`SCHEMA_VERSION` stays 8) — this is
pure DRY. Two deliberately-accepted cosmetic shifts: (1) `read_finding` /
`edit_finding` / `delete_finding` / `read_chunks --around` now attach
`foreign_finding_ids` to their `MEMORY_OFF` error too (previously only
`merge_findings` did) — additive, more informative, breaks no test, and it falls
out of a single uniform predicate rather than a per-site flag; (2) the
`--around-chunk` `MEMORY_OFF` message is now the standard finding-centric wording
("finding N was authored by another session") rather than the old chunk-centric
phrasing. The `MEMORY_OFF` *code* and `merge_findings`'s `foreign_finding_ids`
field — the only things any test asserts — are preserved exactly.

**Supersedes** the per-script-gate stance that GH-0056 / GH-0271 / GH-0272 /
GH-0275 took for the #276 sweep. That stance was a deliberate conflict-avoidance
measure — a `_common.py` predicate would have made all the sub-issue gating
patches touch one file and collide on the omnibus branch (#276 "Optional
consolidation" note). With every sub-issue already merged to `omnibus/v0.8.7`,
that reason is gone: this change is the only one touching the shared chokepoint,
so it carries no conflict. The chokepoint is now the source of truth; the wall is
no longer "as strong as its least-careful copy."
