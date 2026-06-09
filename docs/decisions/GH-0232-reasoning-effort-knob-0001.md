# GH-0232 — Reasoning effort is a unified enum mapped per-provider, via the modern Anthropic effort API

Issue #232 asked for a configurable reasoning-effort knob so corpus-scale
summarization stops paying for default-effort reasoning tokens (the dominant
cost on `gpt-5-nano`). We expose **one config key, `reasoning_effort`**, an enum
`minimal | low | medium | high` (default `low`), threaded through
`Provider.summarize` → both cloud providers. It applies to `summarize()` only —
not `classify()`/`analyze_image()` — since summarization is the cost driver and
the others are low-volume.

## Why this shape, and where it diverges from the issue

The issue was drafted against Anthropic's **old** extended-thinking API
(`thinking={"type":"enabled","budget_tokens":N}` plus a temperature-vs-thinking
guard). That API no longer applies to current Claude models, so we did **not**
implement it as written:

- **`budget_tokens` is removed on Opus 4.7/4.8** (returns 400). The modern,
  GA control is `output_config={"effort": "low|medium|high|max"}` — and it
  happens to mirror OpenAI's `reasoning_effort` string, so a single effort enum
  maps cleanly to both providers instead of needing a token-budget translation.
- **We send effort *without* a `thinking` block.** Anthropic rejects an
  enabled-thinking request that also forces a specific tool, and the Anthropic
  summarize path forces `save_summary` via `tool_choice`. Effort alone still
  lowers reasoning spend (the goal), and dropping the thinking block sidesteps
  that incompatibility entirely.
- **Effort is model-gated.** `output_config.effort` 400s on Sonnet 4.5 / Haiku
  4.5 and earlier — and Haiku 4.5 is this repo's *default* Anthropic model. So
  `AnthropicProvider.summarize` only sends the knob for an allowlist of
  effort-capable prefixes (Opus 4.5/4.6/4.7/4.8, Sonnet 4.6); on every other
  model it leaves behavior unchanged. The allowlist is a hardcoded prefix tuple,
  accepted as brittle-but-simple over a per-ingest Models API capability lookup.
- **Temperature is dropped only where it 400s.** On Opus 4.7/4.8 `temperature`
  is removed, so on the effort path for those models we omit it (and warn once,
  mirroring the OpenAI shim from [GH-0222](GH-0222-omit-temperature-openai-0001.md)).
  On the older effort-capable models temperature is still accepted, so we keep
  forwarding it for determinism. `minimal` has no Anthropic equivalent and maps
  to `low`.

OpenAI is the clean path: `reasoning_effort` is passed verbatim to
`chat.completions.parse`, and only when configured, so non-reasoning models are
never handed an unknown parameter. Ollama and wsjpt accept the parameter and
ignore it — neither has a reasoning-effort concept.

Additive, non-schema change: `reasoning_effort` is a new nullable config key
written by the `bartleby config` wizard. No `SCHEMA_VERSION` bump, no re-ingest.
