# In-run duplicate files are dropped with a surfaced count, not aliased or provenance-tracked

Issue #225. Two byte-identical files (same SHA-256 → same `file_hash`) in one
ingest run used to crash on `documents.file_hash`'s UNIQUE constraint. The fix
deduplicates within the run (`_classify` queued-hash bucket) and reuses the
existing row at the write site (`persist_parse`). That left an open question:
when the second file is detected as a duplicate, what do we do with it?

Three options were on the table:

- **(a)** Drop it, report a count (`Skipping <name> (duplicate content within
  this run)`), record nothing.
- **(b)** Record it somewhere as "duplicate-of `<document_id>`" for provenance.
- **(c)** Add the duplicate's *filename* to the surviving document as an alias.

## Decision

Go with **(a)** — drop and count.

It is the smallest fix that resolves the crash, needs no schema change, and
keeps the existing reporting shape (it mirrors the `skipped` already-ingested
tally right beside it). (b) and (c) both want a place to put data the schema
doesn't have today, which is more than a bugfix should carry.

## The tension, recorded so we don't forget it

Dropping the duplicate's *name* is not obviously free. The
corpus-filename-mislabeling history (ethics-agreement corpus, finding #22) shows
filenames can carry signal a content hash can't — the same bytes filed under two
different names may mean two different things to a researcher. So (b)/(c) remain
legitimate follow-ups *if* provenance of dropped duplicates turns out to matter
in practice. We are deferring them, not rejecting them; revisit if a corpus
surfaces a case where the dropped filename was load-bearing.
