# chunks-chokepoint guard test

The polymorphic-chunks invariant (all writes to `chunks`/`chunks_fts`/`chunks_vec`
go through `bartleby/db/chunks.py`) was enforced only by code review. `tests/test_chunks_chokepoint.py`
now statically walks `bartleby/**/*.py` and fails on any `INSERT INTO` targeting
those tables outside the allowlisted helper module.

Static text scan (not AST): the helper's INSERTs live inside SQL string literals,
so stripping strings would hide the very statements we allow. The matcher uses a
`(?![\w.])` right boundary so `chunks` can't false-match a prefix of
`chunks_archive`/`chunks_fts`/`chunks_vec`, and only fires on `INSERT INTO`
(not SELECT/DELETE or prose mentions). Allowlist is `bartleby/db/chunks.py` alone.
