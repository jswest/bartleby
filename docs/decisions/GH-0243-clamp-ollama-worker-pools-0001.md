# A local Ollama provider auto-clamps the caption/summarize worker pools to 1.

Ingest fans out two post-parse LLM stages across thread pools — captioning
(`caption_workers`, default 4; [GH-0166]) and summarization (`summarize_workers`,
default 4; [GH-0188]). Both defaults were written for a rate-limited cloud
provider that genuinely processes a few requests in flight. For a local **Ollama**
provider they do nothing: Ollama's `OLLAMA_NUM_PARALLEL` defaults to **1**, so a
single model serializes requests, and raising it isn't viable for Bartleby's
default `qwen3-vl:30b` on one GPU — required RAM scales by
`OLLAMA_NUM_PARALLEL × OLLAMA_CONTEXT_LENGTH`, so parallel slots need multiples of
the model's context resident at once. The 4 worker threads pointed at one Ollama
model just queue against a serializing server: no throughput, and they hold 4
inputs in flight for nothing. The consts comments had long *admitted* "a
single-GPU local Ollama serializes anyway" without acting on it.

**The decision:** make worker resolution provider-aware. `_resolve_caption_workers`
returns 1 when `vision_provider == "ollama"`; `_resolve_summarize_workers` returns
1 when `provider == "ollama"`. Cloud providers (anthropic / openai / wsjpt) keep
today's behavior — default 4, explicit override honored.

**A hard clamp, not a clamped default.** Ollama returns 1 *regardless* of a
configured count; an explicit `caption_workers` / `summarize_workers` > 1 is
ignored with a one-line warning rather than honored. The escape hatch for an
exotic operator who raised `OLLAMA_NUM_PARALLEL` on a small model with spare VRAM
is deliberately *not* preserved — the issue (#243) is "remove the parallelization
cruft for Ollama," and a clamp that an override could defeat isn't a removal. The
guiding asymmetry: the parallelism it removes was already inert for the
single-GPU case the default targets, so the cost of the hard clamp is borne only
by a configuration that barely exists.

The `bartleby config` wizard mirrors this: when a stage's provider is Ollama it
skips that stage's worker-count prompt and pops the key, so an Ollama config
simply carries no count and the resolver's clamp supplies the 1. Parse-worker
sizing (`max_workers`) is untouched — it's RAM-bound and provider-agnostic, sized
against measured peak RSS ([GH-0236]), not the LLM backend.

No schema change; tuning runtime worker resolution, no re-ingest. Additive to the
v0.8.4 patch bundle (#237).

[GH-0166]: GH-0166-decouple-captioning-own-concurrent-phase-0001.md
[GH-0188]: GH-0188-parallelize-summarize-pass-0001.md
[GH-0236]: GH-0236-tune-per-worker-gb-to-measured-peak-0001.md
