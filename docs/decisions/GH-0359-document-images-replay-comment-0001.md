# GH-0359 — `document_images` replay comment correctly names the guard

The old `INSERT OR IGNORE INTO document_images` comment in `bartleby/ingest/writer.py` claimed the `OR IGNORE` dedupes replays of the `(doc, image, page, index)` tuple across runs. That is false in general: `document_images`' composite primary key includes the nullable `page_number` (`schema.py`), and SQLite treats NULLs as distinct in a unique/PK index. docling/HTML/standalone images carry `page_number = None` (`parsers.py`), so for those rows the `OR IGNORE` dedup is inert — it never fires.

The actual replay guard is the document-exists early return (`writer.py`, `document_id_for(file_hash)` → return existing `document_id`): a re-run of a byte-identical file returns before reaching the image loop. The `OR IGNORE` is only a secondary belt-and-braces guard, and it covers solely non-NULL `(doc, image, page, index)` tuples.

Comment-only change; no behavior, SQL, or schema change. The corrected comment now states the early return is the load-bearing replay guard and scopes `OR IGNORE` to non-NULL tuples. Part of the v0.9.0 additive schema-bump omnibus (#363).
