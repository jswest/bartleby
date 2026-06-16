# GH-0632 — file-like high-cardinality scope via temp table

**Context.** A broad `--file-like` glob (e.g. `%__202%`) matching ~141k documents
crashed `search`, `scan`, and `list_documents` with `too many SQL variables`. The
previous implementation in `resolve_scope` called `documents_matching_file_like()`,
which materialised all matched document_ids as a Python `list[int]`.  That list then
appeared as `IN (?, ?, …)` bind-parameter lists at three call sites — one in
`Scope.restrict_in` (used by `list_documents`), one in `scan.work`, and one in
`search._build_scope` — each of which hit SQLite's `SQLITE_MAX_VARIABLE_NUMBER`
limit (32,766 on this build) and raised an error.

**Decision.** When `--file-like` is present, `resolve_scope` now materialises the
matched document_ids — intersected with `--in-documents`, `--tag`, and `--authored-*`
date bounds entirely inside SQLite — into a connection-scoped temp table
(`_scope_file_like`) instead of a Python list.  The `Scope` dataclass gains a
`temp_table: str | None` field; when set, `restrict_in()` emits a subquery
(`col IN (SELECT document_id FROM _scope_file_like)`) with no bind parameters.

Three consumer sites were updated:
- `list_documents.py` — already delegated to `Scope.restrict_in`; picks up the fix
  automatically.
- `scan.py` — `_count_match` and `_build_diagnosis` now accept `(restrict_sql,
  restrict_params)` instead of a `restrict: list | None`; the main query in `work()`
  calls `scope.restrict_in("c.source_id")` directly.
- `search.py` — `_build_scope` now accepts the full `Scope` object; when
  `temp_table` is set it returns subquery strings (not Python lists) for each source
  kind.  `_scope_clause` was extended to accept string values (treated as subquery
  SQL) alongside the existing `list[int] | None` values.

**Why.** The fix keeps the match set inside SQLite across the entire request
lifecycle.  No large Python list is ever built; no large bind-parameter list is ever
sent to the engine.  The temp table is `PRIMARY KEY`-indexed so joining against it is
O(log n) per chunk row.  The connection lifetime is the natural cleanup boundary —
the table is dropped implicitly when the connection closes (one per skill invocation).

**Why not a threshold / lazy approach.** A threshold (switch to temp table only above
N ids) would leave the edge case untested on small corpora and require maintaining two
code paths.  Always using the temp table for any `--file-like` invocation is simpler
and consistent.

**Touched files.** `bartleby/skill_scripts/_tags.py`, `scan.py`, `search.py`;
`tests/test_skill_file_like_highcard.py` (new, 33k-document regression fixture).
