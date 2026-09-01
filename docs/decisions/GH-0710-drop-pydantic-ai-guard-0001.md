# Delete the wsjpt `pydantic-ai` version guard and README pin rather than re-pin them. Supersedes GH-0683.

The `#683` guard (`WsjptProvider.__init__` calling `_check_pydantic_ai_version()`)
existed because wsjpt 0.9.1 declared `pydantic-ai>=0.0.45` with no upper bound,
so an unconstrained out-of-band `uv tool install --with` could silently resolve
a `pydantic-ai` incompatible with wsjpt's Vertex/ADC auth path (which called
`GoogleProvider(vertexai=..., project=..., location=...)`, kwargs that only
existed in `pydantic-ai` `[0.3.0, 2.0.0)`). Bartleby compensated with a runtime
preflight check plus a `pydantic-ai>=1,<2` pin in the README install command.

The current wsjpt commit now declares `pydantic-ai>=2.20,<3` itself — its
Vertex/ADC path no longer calls `GoogleProvider(vertexai=...)`, so the kwarg
that motivated the `#683` window is gone. wsjpt pinning its own dependency
properly means the resolver enforces compatibility at install time, exactly
where it should: `uv`'s dependency resolution already fails a bad install
before it ever reaches bartleby's runtime.

This flipped bartleby's guard and README pin from protective to actively
harmful: a correct install (`pydantic-ai>=2.20.0,<3`, satisfying wsjpt's own
declared bound) was then rejected at `bartleby scribe` time by a guard still
enforcing the old `[0.3.0, 2.0.0)` window, and the guard's error message
prescribed a reinstall command (`--with 'pydantic-ai>=1,<2'`) that `uv` could
no longer resolve against the new wsjpt — an unrecoverable loop (issue #710).

**Decision: delete, don't re-pin.** Re-pinning bartleby's guard and README to
the new `>=2.20,<3` window would only recreate this bug the next time wsjpt
moves its `pydantic-ai` requirement without bumping its own version — exactly
what happened between `#683` and `#710`. A bartleby-side mirror of wsjpt's pin
is a copy that goes stale; wsjpt's own `pyproject.toml` bound, enforced by the
resolver at `uv tool install` time, is the single source of truth and doesn't
need a bartleby-side echo.

Removed: `_check_pydantic_ai_version()`, `_PYDANTIC_AI_MIN` /
`_PYDANTIC_AI_MAX_EXCLUSIVE`, the module comment explaining the old window, and
the call site in `WsjptProvider.__init__` (`bartleby/providers/wsjpt.py`); the
four guard tests in `tests/test_providers.py`; the `pydantic-ai>=1,<2` pin from
the README install command and provider table; `re` import (no longer used).
Also fixed the adjacent `ImportError` message in `WsjptProvider.__init__`,
which prescribed a bare `uv pip install 'git+ssh://...'` — a separate `uv pip
install` lands where the running tool can't see it, which is exactly this
issue's failure mode. It now names the correct `uv tool install
'.[docling,sec2md]' --with 'git+ssh://...' --force` command.

Provider and docs only — no `SCHEMA_VERSION` bump (issue #710).

**Residual risk:** nobody has yet seen `bartleby scribe` run end-to-end against
the new wsjpt commit — the reporter's session (tf13, on #710) died at the old
guard, before wsjpt itself was exercised past construction. If wsjpt changed
more than its `pydantic-ai` requirement — e.g. the `Jpt`/`ModelConfig` surface
bartleby calls — that's where it would show. tf13 was asked on the issue to
confirm a real scribe run completes after updating.
