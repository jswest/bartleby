# `ensure_provider_env` stops clobbering an exported `OLLAMA_API_BASE` (issue #490)

> Source: [#490](https://github.com/jswest/bartleby/issues/490)

The anthropic/openai/wsjpt branches of `ensure_provider_env` honor a pre-set env var (`if config_value and not os.environ.get(env_var)`), matching the docstring's "(without overwriting an existing env var)". The ollama branch was the odd one out: it assigned `os.environ["OLLAMA_API_BASE"]` *unconditionally* — even when no `ollama_base_url` was configured — so an exported `OLLAMA_API_BASE` (e.g. a remote GPU box) was overwritten with the config value, or worse, the `http://localhost:11434` default. `README.md:310` tells users the env var works, but it couldn't whenever `ensure_provider_env` ran first.

**Decision (director, 2026-06-11): take the code fix, not a doc-only line.** The two candidate fixes were (a) edit the env-var branch to mirror its siblings — set `OLLAMA_API_BASE` only when it isn't already present — or (b) leave the clobber and re-document the README/docstring to say "config wins." We took (a). It restores the symmetry the docstring already promises across all four providers, and keeps the precedence the way users (and `OllamaProvider.__init__`, which reads `base_url or OLLAMA_API_BASE or localhost`) already assume: an explicitly-exported env var is the most specific signal of intent and shouldn't be silently discarded. Re-documenting config-wins would have entrenched a surprising special case for the one provider whose base URL users most often point elsewhere.

Resulting precedence for ollama (now identical in spirit to the api-key branches): an exported `OLLAMA_API_BASE` survives untouched; with nothing exported, a configured `ollama_base_url` applies; with neither, the localhost default. `README.md:310` and the `ensure_provider_env` docstring stay accurate **as written** — no edit needed.

Additive, no `SCHEMA_VERSION` bump. Three tests in `tests/test_config.py` pin the contract: exported value survives, configured value applies when unexported, localhost when neither. Out of scope: any change to provider precedence elsewhere or to the other branches.
