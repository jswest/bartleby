# GH-0504 — The Anthropic temperature/effort gates are deny-lists of the older models, so current and future models default to the safe path

> Source: [#504](https://github.com/jswest/bartleby/issues/504) (part of #508)

**Supersedes [GH-0250](GH-0250-unconditional-temperature-drop-opus-47plus-0001.md)**
on the narrow point of *how* the temperature/effort gates are keyed.

GH-0250 dropped `temperature` for an **allowlist** of no-temperature prefixes
(`claude-opus-4-7`, `claude-opus-4-8`) and sent `output_config.effort` for an
**allowlist** of effort-capable prefixes (Opus 4.5–4.8, Sonnet 4.6). Both tuples
were closed: a model released *after* that code — `claude-fable-5` is the live
example — matched neither list, so it kept its `temperature` (a 400 on Fable 5,
the exact #250 regression) and silently lost its effort knob. An allowlist is
fail-unsafe for new models by construction; every Anthropic release would have
to touch this file or reproduce the bug.

This inverts both gates to **deny-list the older models** and default everything
else (current + future) to the safe path. `_KEEPS_TEMPERATURE_PREFIXES` is the
set of older models that still *accept* `temperature` (Opus 4.5/4.6, Sonnet
4.5/4.6, Haiku 4.5, and earlier); `_drops_temperature` is the negation, so any
model off that list — Fable 5, a hypothetical Opus 5 — gets `temperature`
dropped and won't 400. `_NO_EFFORT_PREFIXES` is the set of pre-effort models that
*400 on* `output_config.effort` (Sonnet 4.5, Haiku 4.5); `_supports_effort` is
the negation, so current/future models get effort. The older models' observable
behavior is preserved exactly — Opus 4.5/4.6 and Sonnet 4.6 still take effort and
keep temperature; Sonnet 4.5/Haiku 4.5 still keep temperature and skip effort.
The old allowlist tuples are deleted, not left dormant alongside the new ones.

We deliberately did **not** introduce a `model → params` table or any new
abstraction — that was triaged out as over-engineering for three providers. Two
prefix tuples plus their negations are the smallest fix that flips the default
from fail-unsafe (allowlist) to fail-safe (deny-list).

Separately, `_extract_tool_input` now checks `response.stop_reason`. A
`max_tokens` truncation cuts the response off before the forced tool block lands,
which previously surfaced as the opaque "did not include the tool call" error —
the same message a genuinely tool-less response produces, but with a different
fix (raise `max_tokens`, not retry/debug the prompt). The check raises a
`RuntimeError` that names the `max_tokens` truncation so the cause is legible.

Code-only, no schema change (`SCHEMA_VERSION` stays 9).

**Addendum (omnibus #508 critic loop).** The first cut of the deny-lists stopped
at the 4.5/4.6 generation (`_NO_EFFORT_PREFIXES = (sonnet-4-5, haiku-4-5)`;
`_KEEPS_TEMPERATURE_PREFIXES` = Opus 4.5/4.6 + Sonnet 4.5/4.6 + Haiku 4.5), which
contradicted the comments' own "**and earlier**": still-reachable pre-4.5 models
(`claude-opus-4-0/4-1`, `claude-sonnet-4-0`, the `claude-3*` family, and their
dated snapshot IDs `claude-opus-4-20250514` / `claude-sonnet-4-20250514`) fell
off *both* lists, so they would have had `output_config.effort` sent (a hard 400 —
they predate effort) and lost `temperature` with a misleading warning. The
inversion was right; the enumeration was incomplete. Both tuples now include the
pre-4.5 prefixes plus the two dated snapshot IDs (a bare `claude-opus-4-` prefix
can't be used — it would wrongly swallow 4.5–4.8), verified against the Anthropic
model reference (effort is GA only on Opus 4.5+/Sonnet 4.6/Fable 5; temperature
is rejected only on Opus 4.7+/Fable 5) and pinned by revert-tested parametrized
cases. So the "older models' observable behavior is preserved exactly" claim
above now holds for the *whole* older range, not just the 4.5/4.6 generation.
