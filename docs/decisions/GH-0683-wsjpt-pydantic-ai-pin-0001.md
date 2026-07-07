# `WsjptProvider` preflight-checks the installed `pydantic-ai` version and fails open when it can't tell.

`wsjpt`'s Vertex AI/ADC auth path (the one this repo actually exercises — no
Gemini API key) calls `GoogleProvider(vertexai=..., project=..., location=...)`.
Those kwargs only exist in a narrow `pydantic-ai` window: the unified
`pydantic_ai.models.google`/`providers.google` modules first shipped in 0.3.0
(`<0.3.0` only has the split `google_gla`/`google_vertex` modules and raises
`ModuleNotFoundError`), and the `vertexai` kwarg was removed in 2.0.0 (`>=2.0.0`
raises `TypeError: unexpected keyword argument 'vertexai'`). `wsjpt` 0.9.1
declares `pydantic-ai>=0.0.45` with no upper bound, so an unconstrained
out-of-band install can silently resolve an incompatible version and only
surface the failure per-document, wrapped in the generic `summary failed: ...`
message — opaque enough that the fix (pin `pydantic-ai>=1,<2`, already
documented at `README.md`'s install command) is easy to miss.

`WsjptProvider.__init__` now checks `pydantic_ai.__version__` right after the
`wsjpt` import succeeds and raises a `RuntimeError` naming the required
`[0.3.0, 2.0.0)` range and the exact reinstall command when the parsed
`(major, minor)` falls outside it. The check fails open — returns silently,
no raise — whenever `pydantic_ai` can't be imported or its version can't be
parsed, rather than treating "can't introspect" as "incompatible." This
matters because `pydantic_ai` is not installed in bartleby's own venv (the
test suite injects a fake `wsjpt` module without a real `pydantic_ai`
alongside it), and a real `wsjpt` install always drags in *some*
`pydantic-ai`, so a missing or unparseable version is a signal that the
introspection itself is unreliable, not that the environment is broken. The
pin can't move into `pyproject.toml` either: `pydantic-ai` only arrives
transitively through the out-of-band `wsjpt` install (WSJ-internal git
source, unreachable outside WSJ), so bartleby has no dependency of its own to
pin — a runtime guard plus the README callout is the only lever available.

Provider and docs only — no `SCHEMA_VERSION` bump (issue #683).
