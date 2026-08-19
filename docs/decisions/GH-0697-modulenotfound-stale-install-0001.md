# ModuleNotFoundError gets its own actionable envelope, not a generic one (issue #697)

> Source: [#697](https://github.com/jswest/bartleby/issues/697)

`filetype` became a core dependency (`resolve_extension()` in
`bartleby/ingest/chunk.py`), but a `uv tool install --editable` install only
references the source tree — it picks up new *code* on `git pull` for free,
but never installs a dependency added since the tool env was created. Anyone
who installed before `filetype` landed hit a bare
`INTERNAL_ERROR: ModuleNotFoundError: No module named 'filetype'` from
`save_finding`, with nothing pointing at the actual fix.

Fix is a single seam, same shape as the `USAGE_ERROR` precedent
(`docs/decisions/GH-0402-argparse-json-envelope-0001.md`): `bartleby/skill_runner.py`'s
`run()` already funnels every skill script's failure through one
`try`/`except`. A new `except ModuleNotFoundError` arm sits ahead of the
existing `except Exception` catch-all and emits a distinct `STALE_INSTALL`
code with a message naming the missing module and the fix
(`uv tool install --reinstall`). The JSON envelope shape
(`{"error": ..., "code": ...}` on stdout, non-zero exit) is unchanged — only
the code and message differ from the `INTERNAL_ERROR` catch-all.

**Options considered:**

- **Keep `INTERNAL_ERROR`, just improve the message.** Rejected: the
  codebase already has a convention of distinct codes per recognizable error
  class (`NO_ACTIVE_PROJECT`, `USAGE_ERROR`, `MEMORY_OFF`,
  `DOCUMENT_TOO_LARGE`). A caller that wants to special-case "the
  environment is broken" (e.g. surface a different retry/help path) can't
  distinguish it from an arbitrary bug under one shared code. A new code
  costs nothing and matches how every other recognizable failure class is
  already handled.
- **Probe for missing packages at import time / auto-install.** Rejected —
  explicitly out of scope per the issue and the smallest-fix-that-fits
  agreement. A generic `ModuleNotFoundError` catch at the one wrap site
  covers the whole class of "environment is stale relative to the code" for
  free, without guessing at which packages matter or reaching into the
  user's environment to fix it for them.
- **`e.name` vs. `str(e)` for the module name.** `ModuleNotFoundError.name`
  is set by the real import machinery (i.e. for every actual failed
  `import` in production code) but not guaranteed on a hand-built exception,
  so the message falls back to `str(e)` when `.name` is empty — still names
  the module either way.

Deliberately **not** done: no attempt to catch this earlier (e.g. at CLI
startup) or to special-case which specific packages are "core" — a stale
install can be missing *any* dependency, and the wrap site is the one place
every skill script's failure already funnels through. The README's
"After updating Bartleby" section also got a note (no code change) since the
existing editable-install callout was actively misleading: it told users
they could "skip step 1" without qualifying that skipping only holds for
plain code changes, not new dependencies — the exact trap #697 fell into.
