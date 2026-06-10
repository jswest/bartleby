# The Ollama summarize-worker clamp keys off the *effective* provider, not `config['provider']`.

[GH-0243] made worker resolution provider-aware: `_resolve_summarize_workers`
returns 1 when the LLM provider is Ollama (it serializes — `OLLAMA_NUM_PARALLEL`
defaults to 1, so parallel workers only queue). That decision phrased the gate as
"returns 1 when `provider == 'ollama'`", and the code read it straight from
`config['provider']`.

But `main()` doesn't summarize against `config['provider']` — it resolves the
effective provider as `provider_override or config['provider']`, folding in the
`scribe --provider <name>` CLI override (the same value it already records in the
config provenance snapshot). So the config-keyed clamp disagreed with the run it
was meant to govern: `scribe --provider ollama` over a *cloud* config kept 4
workers hammering a serializing local Ollama (the load-bearing clamp bypassed),
and `--provider anthropic` over an *ollama* config clamped a cloud run to 1 for
no reason. Recorded config (override folded in) and behaviour (override ignored)
contradicted each other.

**The decision (refines [GH-0243]):** `_resolve_summarize_workers` takes the
effective provider as a required `effective_provider` keyword and clamps on that,
never bare `config['provider']`. `main()` computes `effective_provider = provider
or config['provider']` once and threads it into both the snapshot and the clamp,
so the two can't drift. The clamp itself is unchanged — still a hard clamp to 1,
explicit `summarize_workers > 1` ignored with a warning ([GH-0243]); only the
provider it tests is now the one summaries actually run against.

Scope is the summarize stage only. Captioning's `vision_provider` has no CLI
override (`_resolve_vision_provider` reads config alone), so its clamp can't
disagree with the run and is left as-is.

No schema change; runtime worker resolution, no re-ingest. Additive to the
v0.8.10 omnibus (#315).

[GH-0243]: GH-0243-clamp-ollama-worker-pools-0001.md
