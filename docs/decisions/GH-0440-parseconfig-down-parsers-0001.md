# Pass ParseConfig down the per-format parse helpers (issue #440)

> Source: [#440](https://github.com/jswest/bartleby/issues/440)

`ParseConfig` — the frozen, picklable dataclass that already crosses the spawn
boundary into `_parse_request` — was being **exploded into seven scalars**
(`pdf_converter`, `html_converter`, `sparse_text_threshold`,
`ocr_min_confidence`, `vision_enabled`, `vision_max_dimension`,
`vision_min_dimension`, `archive_root`) and re-threaded by hand through
`_parse_document`, `_parse_pdf_pdfplumber`, `_parse_pdf_docling`,
`_parse_image_file`, and `_parse_image_routes`. The same keyword block was
repeated across six signatures and four internal call sites; `_parse_document`
alone carried a 17-line signature. Every scalar was only ever sourced from the
one `ParseConfig` instance.

Fix: those five helpers now take the `ParseConfig` object as a positional
`config` argument (alongside identity fields `file_hash`/`file_name` and the
`on_stage`/`on_warn` callbacks) and read `config.vision_max_dimension` etc.
directly. `_parse_request` passes `config` through **unexploded**. No per-format
helper's signature exceeds `(path, config, identity, callbacks)`.

After this change the per-format parse helpers receive the run-wide
`ParseConfig` object instead of seven exploded scalars. Nothing is no longer
handled: every value still originates from the same frozen dataclass that
already crosses the spawn boundary into `_parse_request`, the helpers remain
DB-free, and no new pickling surface is created (the config object is already
present in the worker process when these functions run). The only thing given up
is that each helper's signature no longer documents exactly which settings it
consumes — acceptable in a module where five functions threaded the same seven
parameters and the signature noise now outweighs that documentation value.

**No new dataclass or abstraction.** `ParseConfig` is the existing,
already-present spawn-boundary payload; this change only stops disassembling it.
The tests that constructed the scalar block by hand now build a `ParseConfig`
through a small test-local `_parse_config(...)` factory (defaults for the
run-wide fields, overrides only for the `archive_root`/`vision_enabled`/
`vision_min_dimension` that actually vary across sites). Behavior is unchanged;
`uv run pytest` stays green.

---
*Filed from the 2026-06-11 dry sweep (dead/wet/bloat audit; every item
adversarially verified by an independent defender pass).*
