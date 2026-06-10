# Mutation tests assert chunks / chunks_fts / chunks_vec stay in sync — checked per-leg (issue #344)

> Source: [#344](https://github.com/jswest/bartleby/issues/344)

The skill mutation paths (`edit_finding`, `merge_findings`, `save_summary`
replace, `delete_finding`) rewrite the polymorphic `chunks` table and its two
mirrors — the `chunks_fts` external-content FTS5 index and the `chunks_vec`
`vec0` table — through the typed `bartleby.db.chunks` helpers. The tests asserted
the `chunks` row deltas but only spot-checked the mirrors, and where they touched
`chunks_fts` they did so **vacuously**. This issue adds one shared assertion and
closes the specific gaps. It is **tests-only**: no production code path changed,
no `SCHEMA_VERSION` bump.

**The two mirrors must be checked DIFFERENTLY — that asymmetry is the point.**
`assert_chunk_tables_consistent(conn)` lives in `tests/_skill_fixtures.py` (where
the suite already shares fixtures/helpers) and verifies each leg with the only
check that is *non-vacuous* for that leg:

- **FTS leg → FTS5's own `'integrity-check'`.** `chunks_fts` is declared
  `content='chunks'` (`schema.py` ~158-164), so it stores no row text of its own
  — any rowid lookup or `COUNT` over it is satisfied by reading *through* `chunks`
  and therefore can never observe index/content drift. The pre-existing
  `SELECT … FROM chunks_fts WHERE rowid = ?` assertions in `delete_finding` and
  `edit_finding` were exactly this vacuous shape. The only thing that catches a
  drifted external-content index is `INSERT INTO chunks_fts(chunks_fts)
  VALUES('integrity-check')`, which raises if the FTS index disagrees with its
  content table.
- **Vector leg → direct rowid-set comparison.** `chunks_vec` is a real `vec0`
  table with its own row storage, so `{rowid FROM chunks_vec}` must equal
  `{chunk_id FROM chunks}` exactly, and a mismatch is a true observation.

The helper is called at the END of the `edit_finding` body-rebuild,
`merge_findings` happy-path, `save_summary` replace, and `delete_finding`
mutation tests. The vacuous per-rowid `chunks_fts` checks in `delete_finding`
(and the FTS-only mirror check in `edit_finding`) are replaced by the
integrity-check call; the vec leg keeps explicit per-id presence/absence
assertions because the objective asks each mutation to name the specific chunks
that should land in / leave `chunks_vec`.

**`chunk_id` (and `summary_id`) rowid reuse forced identity-by-content, not
by-id.** SQLite reuses freed `INTEGER PRIMARY KEY` rowids, so after a
delete-old/insert-new dance the *replacement* chunks frequently re-own the freed
ids. Naive "the old id is gone" assertions therefore pass vacuously (or fail
spuriously). The gap-closers account for this:

- **`save_summary` replace.** The fixture's prior summary for doc A was seeded
  with **zero** chunks, so the replace's delete-old/insert-new was unobserved.
  The seed now inserts two prior summary chunks (`_skill_fixtures.py`), and the
  test identifies the old summary by its **distinctive chunk text** (not its
  `summary_id`, which the replacement re-owns) and asserts those texts are gone
  from `chunks`, that the surviving summary's chunks reflect the new body, and
  that all three tables agree. A new seed chunk count broke one unrelated test
  (`test_skill_search.py::test_search_with_summaries_source_kind` seeded its own
  summary chunk at `chunk_index=0`, now colliding on the
  `(source_kind, source_id, chunk_index)` UNIQUE) — bumped its `chunk_index` to
  2; that is the only edit outside the four mutation-test files and the fixtures.
- **`edit_finding` body rebuild.** Captures the finding's body-chunk ids before
  the edit, then asserts the new ids are present in `chunks_vec` and any old id
  **not reused** by the new body is absent (today the test checked FTS but never
  vec).
- **`merge_findings` happy path.** Captures the source findings' body-chunk ids
  before the merge, then asserts each source chunk not re-owned by the rebuilt
  target body is absent from both mirrors and the target's new chunks are present
  in both.

**The assertions pass on current HEAD — no production bug surfaced.** Per the
issue, in the happy path these operations *should* keep the three tables
consistent because they go through the typed helpers; a failure would have been a
real write-path bug to PARK and report, not fix here. None failed.

touches: `tests/_skill_fixtures.py`, `tests/test_skill_edit_finding.py`,
`tests/test_skill_merge_findings.py`, `tests/test_skill_save_summary.py`,
`tests/test_skill_delete_finding.py`, and `tests/test_skill_search.py` (the
one-line collision fix forced by the richer seed). No product code, no schema
change (`SCHEMA_VERSION` unchanged).
