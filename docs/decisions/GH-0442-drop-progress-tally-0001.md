# Pass the progress phase handle directly to caption/summarize (issue #442)

> Source: [#442](https://github.com/jswest/bartleby/issues/442)

The caption (#166) and summarize (#188) passes drove their progress through a
double adapter that existed only to translate between two shapes nobody else
consumed. The chain: `_caption_all`/`_summarize_all` wrapped a `_ProgressTally`
(in `bartleby/ingest/progress.py`) that converted `advance()` into
`on_progress(done, total)`; `scribe.py` passed `cap_phase.on_progress` /
`sum_phase.on_progress`, where `_Phase.on_progress` converted `(done, total)`
**back** into `start(total)` / `set_completed(done)`; `set_completed` and
`ScribeProgress._set_completed` existed solely to serve that round-trip. Each
phase additionally threaded a second separate callback, `on_lane`, which was just
`phase.lane`. The only production consumer of the `(done, total)` contract was
`ScribeProgress` itself — `_ProgressTally`, the `on_progress`/`on_lane` parameter
pairs, and the `_Phase.on_progress` re-adapter appeared nowhere else in
`bartleby/`, `scripts/`, `benchmarks/`, or `web/`.

Fix: `_caption_all` and `_summarize_all` now take `phase: _Phase | None` instead
of `on_progress` + `on_lane`. They call `phase.start(len(pending))` once and
`phase.advance()` per item, guarding every call on `phase is not None` so the
headless path (tests, and any future caller that wants no display) is a clean
no-op. `scribe.py` passes `progress.phase("caption")` / `progress.phase(
"summarize")` directly. Deleted: `_ProgressTally`, `_Phase.on_progress`,
`_Phase.set_completed`, and `ScribeProgress._set_completed`. The parse phase
already drove its handle this way (`parse_phase.start`/`advance`), so all three
phases are now uniform.

**Rendered progress output is unchanged.** The old chain was a pure round-trip
onto the same three primitives the handle now calls directly:

- The reveal — `_ProgressTally.__init__` fired `on_progress(0, total)`, which
  `_Phase.on_progress` routed to `start(total)`. Now `phase.start(len(pending))`
  is called once at the same point, so the denominator is revealed identically
  (header tally flips from `—` to `0/total`, the overall bar grows by `total`).
- Per-item ticks — `_ProgressTally.advance()` fired `on_progress(done, total)`
  with monotonically rising `done`, which `_Phase.on_progress` routed to
  `set_completed(done)`, i.e. `self._done[name] = done`. Since `done` rose by
  exactly 1 each tick, that is identical to `_advance(name, 1)`, which the handle
  now calls. Both paths land on the same `self._done[name]`, run the same
  `_recompute_eta`, and call the same `_refresh_overall` — so the header tally,
  the ETA, and the overall bar render bit-for-bit the same frames.
- Lanes — `on_lane=cap_phase.lane` became `phase.lane(...)`; same method, same
  arguments, same call sites.

## What is no longer handled

The caption and summarize phases no longer expose a renderer-agnostic numeric
progress contract (a bare `Callable[[int, int], None]` plus a bare lane callable)
that an alternate consumer could plug into; they now take the `ScribeProgress`
phase handle (or `None`). That is fine because in the code's entire history only
`ScribeProgress` ever consumed these callbacks — the `(0, total)`-reveal
convention, the `_ProgressTally` "shared meter", and the `on_progress`/`on_lane`
parameter pairs existed purely to adapt between the phases and that one renderer,
and the parse phase already used the handle directly. If a second progress
consumer ever appears, the callback seam is trivially re-introduced at that point;
carrying a two-way adapter today is speculative generality the repo's culture
explicitly rejects.

**No new abstraction.** This deletes an adapter and points the callers at a handle
that already existed; no layer is introduced. `uv run pytest` stays green — the
direct-call tests in `tests/test_scribe.py` now pass a small `_StubPhase` (records
`start`/`advance`/`lane`) and assert against it instead of against the reconstructed
`(done, total)` tuples; `tests/test_progress.py` exercises the overall bar via
`advance()` rather than the deleted `set_completed`.

---
*Filed from the 2026-06-11 dry sweep (dead/wet/bloat audit; every item
adversarially verified by an independent defender pass).*
