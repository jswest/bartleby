# Reuse one wsjpt `Jpt` per (schema, model, instructions), guarded by a lock (issue #755)

> Source: [#755](https://github.com/jswest/bartleby/issues/755)

**The bug.** `WsjptProvider.summarize`/`classify`/`analyze_image` each built a
fresh `wsjpt.Jpt(...)` per call. `Jpt.__init__` calls `_create_model()`, which
builds a new `genai.Client` / `GoogleProvider` and thus a new HTTP connection
pool. Nothing closed the old ones. Over a scribe run with thousands of caption
calls, open sockets piled up past macOS's default 256-fd limit and every
subsequent call failed with `[Errno 24] Too many open files`.

**The fix.** `WsjptProvider` now caches `Jpt` instances on `self._jpt_cache`,
keyed by `(schema, model, custom_instructions)` — the three inputs that
actually vary across `summarize`/`classify`/`analyze_image` calls. A
`threading.Lock` guards cache-miss construction so two threads racing on the
same key can't each build (and leak) a redundant `Jpt`. This mirrors the
existing precedent: the Anthropic, OpenAI, and Ollama providers each build one
client in `__init__` and reuse it for the provider's lifetime.

**Why sharing a `Jpt` across threads is safe here.** Caption and summarize
both run through 4-thread pools, so this needed checking against wsjpt's
source (`jpt.py`, `_background_loop.py`), not just assumed:

- `Jpt.parse()` runs its async work via `wsjpt._background_loop.run_sync`,
  which submits every coroutine — from every `Jpt` instance, in the whole
  process — onto one lazily-started, module-global background event loop
  thread. Concurrent `parse()` calls from bartleby's thread pool interleave as
  coroutines on that one loop (they don't serialize), and the shared
  `genai` async client stays bound to it — safe for concurrent requests.
- `Jpt` does carry per-call mutable instance state (`last_parse_duration`,
  `last_retry_count`, and a set of `_auto_tag_*` dicts mutated during
  AutoTag-field processing). Bartleby never reads `last_parse_duration` or
  `last_retry_count`, so races on those are inert for bartleby — though wsjpt's
  own `parse_history` rows can record another call's duration/retry count. The `_auto_tag_*` state is
  only touched when the parsed Pydantic model declares AutoTag-annotated
  fields (`wsjpt.autotag.discover_auto_tag_fields`); bartleby's schemas
  (`DocumentSummary`, `VlmDescription`, and the tag-classification schemas in
  `_tags.py`) declare none, so that code path is never exercised by this
  provider — confirmed by grep, no `AutoTag` usage anywhere in `bartleby/`.
- `custom_instructions` is read fresh on every `parse()` call but is fixed at
  `Jpt` construction by this provider (never reassigned afterward), so it
  can't drift between callers sharing the same cached instance.

Given those three points, a single shared `Jpt` per key is safe for
bartleby's usage; a per-thread `Jpt` or close-after-each-call alternative
would have been unnecessary extra complexity.

No schema change, no `SCHEMA_VERSION` bump. `bartleby/providers/wsjpt.py`
only.
