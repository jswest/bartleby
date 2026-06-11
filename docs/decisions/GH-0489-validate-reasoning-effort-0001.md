# reasoning_effort is validated at the scribe chokepoint — hard-fail, no bare KeyError (issue #489)

> Source: [#489](https://github.com/jswest/bartleby/issues/489)

`_EFFORT_MAP[reasoning_effort]` was a bare dict lookup on a value read straight
from `yaml.safe_load` with no validation outside the interactive wizard. A
hand-edited config with e.g. `reasoning_effort: xhigh` crashed mid-ingest with a
raw `KeyError` on the anthropic provider, and was forwarded verbatim to OpenAI.
Neither failure named the offending config field.

**Director decision (2026-06-11): HARD-FAIL.** An out-of-enum value is rejected,
not silently coerced to the default effort. This bundle's theme is "expected
error → clean exit (1) on stderr, never a traceback," and a silent fall-back
would mask a config typo the user meant to act on.

Fix is a single seam where scribe reads the config, reached **before any
provider call** — so it protects every provider (anthropic's `_EFFORT_MAP`
lookup *and* the OpenAI/wsjpt/ollama forwards) at once, not just the anthropic
path. `scribe.main()` already reads `config.get("reasoning_effort")`; a check
right there rejects an out-of-enum value via `console.error(...)` + `sys.exit(1)`
— the same clean exit-1-on-stderr pattern used by the sibling expected-errors in
`commands/project.py`, `embed.py`, `ready.py`, and `serve.py`. The message names
the field, the bad value, and the allowed set
(`minimal, low, medium, high`). No traceback.

The allowed enum (`ALLOWED_REASONING_EFFORTS`) and its default
(`DEFAULT_REASONING_EFFORT`) moved from `commands/config.py` into
`bartleby/lib/consts.py`, where the other shared `ALLOWED_*` lists already live.
That module imports nothing, so scribe and the wizard now both source the enum
from one place — the values scribe rejects can never drift from the ones the
wizard offers. The wizard keeps constraining the value at input time as before;
this gate covers the hand-edited path it can't see.

Defense in depth: the anthropic provider's raw `_EFFORT_MAP[...]` lookup became
`_map_effort(...)`, a `.get` that raises a `ValueError` naming the field instead
of a bare `KeyError`. scribe rejects bad values before the provider is ever
reached, so this is a backstop for a caller that drives the provider directly —
the issue's suggested `_EFFORT_MAP.get(...)` form, kept as a second layer rather
than the primary gate.

Deliberately **not** done: no global try/except added to `cli.py` (the existing
converter-validation `ValueError`s in `scribe.main()` predate this and aren't in
scope); no schema change (the value lives in `config.yaml`, not the DB); no new
abstraction beyond the one `_map_effort` helper.

Tests: `tests/test_scribe.py` asserts an out-of-enum `reasoning_effort` exits 1
with the field named on stderr (captured via the repo's `console.error`
monkeypatch idiom, since the shared Rich Console binds its stream at import
time), plus a positive control that a valid value ingests. `tests/test_providers.py`
covers the anthropic `_map_effort` `ValueError` backstop. No schema change.
