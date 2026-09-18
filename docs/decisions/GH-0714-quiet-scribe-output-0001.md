# Aggregate the per-image skip line; close the parse pool instead of terminating it.

`bartleby scribe` had four sources of noise on an otherwise clean run (issue
#714): a pydantic-ai/Logfire banner, dozens of near-identical "Skipping image
… below the 256px vision minimum" lines, a docling preset-registration log
line repeated once per worker warmup, and a `resource_tracker` `UserWarning`
about leaked loky semaphores printed after the shell prompt at shutdown. None
of the four are errors; all are either one-time setup chatter or a byproduct
of how the parse pool tears down.

**Banner and docling preset log — silence at the source, unconditionally.**
`bartleby/providers/wsjpt.py` now sets `PYDANTIC_AI_NO_BANNER=1` (via
`setdefault`, so a user who's wired up Logfire is unaffected) before the
lazily-imported `wsjpt`/`pydantic_ai` ever loads. `bartleby/lib/quiet.py`'s
`setup_quiet_third_party` now also pins
`logging.getLogger("docling.datamodel.stage_model_specs")` to `CRITICAL`,
ahead of the existing `if verbose: return` early-out — this docling log line
(`_log.error(...)` on a harmless, idempotent-by-design preset
re-registration) is never actionable in any mode, so it isn't gated by
`--verbose` the way the rest of that function's quieting is. `setup_quiet_third_party`
already ran in every worker's `_init_worker` and in the main process before
the inline (`max_workers <= 1`) path, so no new call site was needed.

**Per-image skip spam — aggregate per document, per-image detail moves behind
`--verbose`.** `bartleby/ingest/parsers.py`'s `_parse_image_routes` used to
call `on_warn` once per undersized image; a document full of logos, table
rules, or equation snippets could produce dozens of near-identical lines. It
now counts skips and, when the count is > 0, emits one line ("Skipped N
undersized image(s) (below the <px>px vision minimum)."). The issue's fallback
plan was "just do the aggregate line" if verbose plumbing wasn't cheap — it
turned out to be one field. `ParseConfig` gained a `verbose: bool = False`
field (`bartleby/commands/scribe.py` already had `verbose` in scope where it
builds the `ParseConfig`, so wiring it through was a single added kwarg, not
new plumbing), so `--verbose` still restores the old per-image page +
dimensions detail instead of the aggregate line. What gets skipped is
unchanged — only what gets printed, and only when a document actually has
undersized images.

**Leaked loky semaphores — close/join the pool on the happy path, terminate
only on failure.** `bartleby/ingest/pool.py`'s `parse_stream` used
`with ctx.Pool(...) as pool:`, whose `__exit__` calls `terminate()`
unconditionally — SIGTERMing every worker as soon as the last result is
yielded. A SIGTERMed worker never runs its own atexit cleanup, so
docling's transitive joblib/loky executor (pulled in via scikit-learn) leaks
its named semaphores, and Python's `multiprocessing.resource_tracker`
complains at interpreter exit — after the shell prompt has already returned,
looking like an error from a finished, successful run. The pool construction
was pulled out of the `with` and the pool is now driven by an explicit
`try/except BaseException/else/finally`: the happy path (the `else` clause,
reached only once `imap_unordered` is fully drained without exception) calls
`pool.close()` then `pool.join()`, letting workers exit on their own; any
exception during the drain still hits `pool.terminate()` so a failing run
tears down promptly rather than waiting on workers that may be stuck. The
existing `finally` block (stopping the progress-drain thread and shutting
down the `Manager`) is unchanged. Amended on leaf-critic review: the `ctx.Pool(...)` call itself sits *inside* that outer `try`, as it did under the `with`, so a failed construction (spawn exhaustion, too many open files) still reaches the drain/`Manager` cleanup rather than leaking the `Manager` process. `BaseException` (not `Exception`) is
deliberate: if the generator itself is abandoned mid-iteration (e.g. a caller
stops consuming `parse_stream` early), Python throws `GeneratorExit` at the
suspended `yield from`, and that path should terminate rather than block
trying to join workers that may still be mid-task.

**Not verified in this environment:** wsjpt isn't installed in the dev
sandbox, so the banner suppression couldn't be exercised end-to-end; the fix
is a `setdefault` landing before the lazy `wsjpt` import, matching the
documented pydantic-ai behavior. The loky semaphore leak and its fix are also
unverified beyond the pool restructuring being reviewed against the
documented `Pool.__exit__`/`terminate()`/`close()`/`join()` semantics — a full
repro needs the docling+scikit-learn dependency chain and multiple worker
processes exercising a real ingest, which the guardrail on this issue
excludes running against a real corpus.
