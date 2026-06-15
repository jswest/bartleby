# Separate `vision_temperature` knob, defaulting to summarize `temperature` (issue #223)

> Source: [#223](https://github.com/jswest/bartleby/issues/223)

Captioning previously rode the model's *default* temperature: `analyze_image` never sent the parameter at all, so a corpus that set `temperature` for deterministic summaries had no say over caption determinism. #223 adds a dedicated `vision_temperature` config key threaded the same way `temperature` is threaded through summarization. **Additive, non-schema** — no `SCHEMA_VERSION` bump, no re-ingest; the value is read at ingest time and never persisted to a column.

**Default-to-summarize, resolved at one point.** When `vision_temperature` is unset it falls back to the summarize `temperature` (then to `DEFAULT_TEMPERATURE`): `config.get("vision_temperature", config.get("temperature", DEFAULT_TEMPERATURE))`. The resolution lives in `commands/scribe.py` alongside the existing `temperature` resolution — the single point where ingest config becomes runtime values — and is threaded down through `caption._caption_all` → `caption._analyze_image` → `images.analyze` → `provider.analyze_image`. One knob keeps an unconfigured corpus behaving exactly as before (caption temperature == summarize temperature) while letting a corpus that wants deterministic summaries but more varied captions split the two.

**Provider parity with the summarize path, not new policy.** `analyze_image` gained a required `temperature: float` on the `Provider` protocol. The per-provider handling deliberately *reuses the summarize-path rules* rather than inventing vision-specific ones, because the temperature-acceptance quirks are properties of the model, not of the call:
- **anthropic** reuses the existing `_drops_temperature` / `_warn_dropped_temperature` helpers — Opus 4.7+/Fable-5 (and future models) 400 on `temperature`, so it's dropped there and forwarded on the older deny-listed models, identical to `summarize`/`classify`.
- **openai** reuses `_drop_temperature` — GPT-5 models accept only the default (1.0); we never send it and warn once.
- **ollama** forwards it via `options={"temperature": ...}`, as `summarize` does.
- **wsjpt** accepts the param but **ignores** it (kept out of the signature would break the protocol). wsjpt owns model settings centrally, so callers can't drift from toolkit defaults — same stance it already takes for `summarize`/`classify`.

No separate "vision temperature can be a different range" validation — the wizard's `_prompt_temperature` (now taking an explicit default rather than reaching into `existing`) already bounds it to [0, 1], and the vision prompt offers the just-chosen summarize temperature as its default so the common case is a single confirmation. The temperature-drop *warnings* are shared module-level once-flags with the summarize path; a config that drops temperature for summaries will already have warned, so captioning doesn't double-warn.
