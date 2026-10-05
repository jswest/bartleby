# `documents.source_path` is stamped by the main-process drain as a `persist_parse` argument, not carried on `ParsedDocument` — and `project info` counts top-level documents only.

Issue #686 adds a nullable `documents.source_path` (schema v12, additive):
the absolute, resolved path of the file handed to `scribe`, so a user coming
back to a corpus can tell where the originals came from. `file_path` keeps
meaning the archive copy. The judgment calls below are about where the path
enters the write and what `project info` counts.

**The path rides `persist_parse(parsed, source_path=...)`, not a new
`ParsedDocument` field.** The source path is not a parse product: no
converter reads or transforms it, and the drain in `ingest/parse.py` already
holds it on `outcome.request.path`. Adding a field would touch every
`ParsedDocument(...)` constructor in `ingest/parsers.py` to forward a value
the converters never use. A keyword argument keeps the change to the one
call site that knows the path. It defaults to `None` (NULL, "unrecorded"),
which is the honest value for any caller that has no source file. The drain
resolves the path in the main process (`req.path.resolve()`), so a relative
`scribe` argument is recorded absolute.

**First-seen wins without any new code.** A byte-identical file is skipped
by `_classify` before it reaches the pool. The resume bucket
(parsed-but-uncaptioned) never writes a `documents` row, and
`persist_parse`'s existing-hash early return guards the write itself. So
nothing can overwrite an existing `source_path`, and no `UPDATE` path was
added.

**`project info` counts top-level documents (`parent_document_id IS NULL`).**
A #254 split filing is one source file but N+1 `documents` rows, all sharing
one `source_path`. Counting every row would inflate a directory's doc count
by the section fan-out, so the "Sources" row and the `--sources` listing both
exclude section rows. The existing "Documents" count is unchanged; it still
counts every row, as before.

**The publish scrub is one `UPDATE` inside `strip_session_layer`'s existing
transaction, not a sibling step.** `source_path` is not session-layer data,
but the function is already "the scrub of a publish copy", and a second
function with its own transaction would add a seam without adding safety.
The docstring says what it now also nulls.
