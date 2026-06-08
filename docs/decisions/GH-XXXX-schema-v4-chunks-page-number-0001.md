# Schema v4 — `chunks.page_number`

first-class `INTEGER` column on `chunks`. Populated for pdfplumber-extracted document chunks (the previous `section_heading = "page N"` hack is gone — section_heading is now `NULL` for pdfplumber). NULL for docling chunks (docling doesn't expose per-chunk pages in our wiring), summaries, findings, and image chunks (image page lives on `document_images`). `search` and `read_chunks` surface it directly; `save_finding` returns it on `citations[]`.
