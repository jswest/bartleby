# Merge the caption/summarize worker resolvers into one (issue #441)

> Source: [#441](https://github.com/jswest/bartleby/issues/441)

`_resolve_caption_workers` and `_resolve_summarize_workers` in
`bartleby/ingest/resolve.py` were two near-identical functions (~71 lines with
docstrings) implementing the **same algorithm**: read a configured count with a
default, let `--timings` force 1 (warning only when the configured value exceeds
1), clamp an Ollama provider to 1 (warning only on an explicit, non-default count
— the #243 serialization clamp), else return `max(1, configured)`. They differed
only in config key (`caption_workers` vs `summarize_workers`), default constant,
provider source (`config['vision_provider']` vs the #314 effective-provider
argument), and the nouns in two warning strings. They had moved in lockstep
through #243 and #314, and their docstrings even cross-referenced each other.

Fix: one parameterized helper,
`_resolve_io_workers(config, *, key, default, provider, verb, unit, timings)`.
`scribe` calls it twice — passing `config.get('vision_provider')` for captions
and the already-computed effective provider for summaries — so each stage threads
in the provider it has already resolved. `verb`/`unit` only shape the two warning
strings, which are now f-string templates that reproduce the prior text exactly.

Nothing functional is no longer handled. The three load-bearing behaviors are
preserved byte-for-byte:

- **`--timings` clamp** — forces 1, warning only when the configured value
  exceeds 1.
- **Ollama explicit-value clamp (#243)** — clamps to 1; the warning re-reads the
  raw `config[key]` (not the default-folded value) so it fires only on an
  explicit count > 1, never on the default.
- **`max(1, n)` floor** — unchanged for the non-clamped path.

The #314 distinction (summaries clamp off the override-folded *effective*
provider, not bare `config['provider']`) is preserved structurally: it now lives
in the call site, which passes the effective name it already computes, rather
than in a per-function branch.

**No new abstraction.** This is a plain helper that collapses two copies of one
algorithm — not a worker-resolution layer. If a future stage ever needs a
divergent clamping rule it can fork back out; an acceptable bet given the two
functions have changed in lockstep through every revision so far. The direct-call
tests in `tests/test_scribe.py` now route through small test-local
`_caption_workers`/`_summarize_workers` wrappers that fix the per-stage
parameters; the assertions (defaults, `--timings`, the Ollama clamp, and the
#314 effective-provider override) are unchanged. `uv run pytest` stays green.

---
*Filed from the 2026-06-11 dry sweep (dead/wet/bloat audit; every item
adversarially verified by an independent defender pass).*
