# Parse workers route warnings to the parent; they never touch the console (issue #227)

> Source: [#227](https://github.com/jswest/bartleby/issues/227)

Parsing runs in a spawn worker pool (`bartleby/ingest/pool.py`, #165). Each
worker is a **separate process** with its own `bartleby.lib.console` `Console`,
which knows nothing about the **parent's** Rich `Live` progress region. So when
worker-side parse code called `console.warn` directly — sub-minimum image skips
and the "no vision provider" notice in `_parse_image_routes`, the inner-document
skip in `_parse_edgar_submission` — the child's raw stderr write interleaved
with the parent's `Live` frames and desynced its anchor: the `Run of show` /
`Overall` header blocks stopped redrawing in place and stacked as fresh lines on
a multi-thousand-file run.

This is **distinct from #208** (which bounded the live region to the
terminal *height* so it didn't self-stack). Same visual symptom, different
cause: out-of-band writes from another process, not region overflow. It only
manifests under the concurrent pool — at `max_workers <= 1` the parse runs inline
in the main process, where `console.warn` coordinates with the `Live` correctly.

**The fix routes worker notices back as data, mirroring how parse *failures*
already return on `ParseOutcome.error`.** A new `on_warn` callback is threaded
parallel to the existing `on_stage` through the `_parse_*` chain; `_parse_request`
collects into a `warnings: list[str]` and returns it on the outcome (on both the
success and the `except` path, so notices collected before a late failure aren't
lost); the main-process drain emits each with `console.warn`, which inserts above
the bar. Threading a second callback alongside `on_stage` was deliberate — it's
the established worker→parent seam in this code, so no new mechanism is
introduced.

**Scope is parse-workers only.** Caption and summarize run in
`ThreadPoolExecutor`s *in the main process* (`scribe.py`), so their `console.warn`
calls — including `images._warn_ocr_degraded_once` (reached from the caption-phase
`images.analyze`) — already coordinate with the `Live` and are left untouched.
The invariant to preserve going forward: **code reachable from `_parse_request`
must not call `console.*`** — it routes through `on_warn`. No schema change —
patch-level.
