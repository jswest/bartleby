# GH-0250 — Temperature is dropped on Opus 4.7+ unconditionally, not only on the effort path

**Supersedes [GH-0232](GH-0232-reasoning-effort-knob-0001.md)** on the narrow
point of *when* the Anthropic provider drops `temperature`.

GH-0232 introduced the `reasoning_effort` knob and, in the course of it,
described dropping `temperature` "only where it 400s … on the effort path for
those models" — i.e. it coupled the temperature drop to whether a
reasoning-effort value was sent. That coupling was wrong in practice: **Opus
4.7+ rejects `temperature` outright** (a 400), and that rejection is a property
of the *model*, not of whether the request also carried an `output_config.effort`
block. A summarize (or classify) call against Opus 4.7/4.8 with effort unset
would still send `temperature` and still 400.

PR #250 (`fix/anthropic-temp-drop`) made the drop **unconditional**: the
Anthropic provider removes `temperature` for any model whose name matches the
no-temperature prefix tuple (`claude-opus-4-7`, `claude-opus-4-8`), independent
of `reasoning_effort`, warning once when a non-default temperature is thereby
ignored. The effort knob and the temperature drop are now orthogonal — effort is
gated by `_supports_effort` (Opus 4.5+/Sonnet 4.6), temperature is gated by
`_drops_temperature` (Opus 4.7+) — and either can apply without the other. Older
effort-capable models (Opus 4.5/4.6, Sonnet 4.6) still accept `temperature`, so
it's kept there for determinism. The same model-keyed drop was later extended to
`classify()` (#256), which has the same exposure.

Everything else in GH-0232 — the unified `minimal|low|medium|high` enum, the
modern `output_config.effort` API, sending effort *without* a `thinking` block,
the model-gated effort allowlist, OpenAI's clean `reasoning_effort` pass-through,
Ollama/wsjpt accepting-and-ignoring — stands unchanged. Only the
temperature-drop *condition* is corrected here.

Code-only, no schema change.
