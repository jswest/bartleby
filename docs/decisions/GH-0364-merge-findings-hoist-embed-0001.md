# Hoist `merge_findings` embed out of the write-lock window (issue #364)

> Source: [#364](https://github.com/jswest/bartleby/issues/364)

`merge_findings` is `run(mutates=True)`, so the runner wraps `work()` in one deferred apsw transaction: no write lock is taken until the first write. Its first write is the `UPDATE findings`, and it called the combined `rebuild_finding_chunks` (embed + write in one) *after* that `UPDATE` — so the ~5–10s lazy sentence-transformers model load ran with the write lock already held. `busy_timeout` is only 5000ms (shorter than the load), so a concurrent skill call, the web UI, or an ingest could hit `BusyError`/`INTERNAL_ERROR`. This is exactly the exposure #340 fixed for `save_finding`/`edit_finding`/`save_summary`; the GH-0340 note wrongly claimed hoisting `merge_findings` would "buy it nothing" on the assumption it already held a write txn across embedding — it did not until the `UPDATE`.

The fix is mechanical and mirrors `edit_finding`: call the existing `embed_body_chunks(body)` (no DB touch) *before* the `UPDATE findings`, and `write_finding_chunks(conn, target, chunk_inputs)` at the SQL tail. The deferred write lock now spans only the millisecond SQL tail, not the model load. No behavior or output change — the same rows are written in the same order; only the embed moved earlier, ahead of the first write.

With this, `rebuild_finding_chunks` in `_common.py` had no remaining callers (it survived in #340 solely so `merge_findings` could keep importing it), so it is deleted per the repo's no-dormant-code rule; `embed_body_chunks` / `write_finding_chunks` remain. The GH-0340 note's "buys it nothing" sentence is corrected in place. No schema change.
