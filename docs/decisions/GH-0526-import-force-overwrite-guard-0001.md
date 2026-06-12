# `bartleby project import` requires `--force` to overwrite an existing project (issue #526)

`project import <name>` adopts a published artifact as local project `<name>`.
The first cut (GH-0510) overwrote a same-named existing project unconditionally:
once the compatibility gates passed it `rmtree`d the project's `archive/` and
replaced its `bartleby.db` with **no confirmation** — unlike `project delete`,
which requires `--yes`. So `bartleby project import my-research --from <url>`
with a name collision silently destroyed a real corpus *and its local findings*,
which a findings-free artifact can never restore. The critic/director pass over
omnibus #494 flagged it as the bundle's one sharp edge and deferred the call.

## Decision

A same-name collision is a **hard refuse unless `--force` is passed**:

- **Without `--force`**, importing into an existing project name raises
  `ImportRefused` (clean non-zero exit, message naming the collision and pointing
  at `--force`). The existing project is left byte-for-byte untouched.
- **With `--force`**, the prior gates-then-overwrite flow runs unchanged.
- A fresh, non-colliding name needs no flag.

The refusal is decided **up front** — before the scratch download and before the
project dir is touched — consistent with import's existing "verify before you
create anything, so a refusal leaves nothing behind" discipline. It costs no
network round-trip and cannot perturb the existing project.

## Why this shape

Considered and rejected:

- **Refuse-always on collision (delete-then-reimport):** safe, but breaks the
  idempotent-re-import contract GH-0510 deliberately built (and a test encodes).
- **Interactive confirm:** import already runs unattended in `/ultraship`
  contexts; a prompt is hostile to automation. A flag composes; a prompt doesn't.
- **Auto-detect "same corpus re-import" vs "different corpus, same name":** would
  need a recorded source identity and extra machinery for a rare case — more than
  the footgun warrants.

`--force` mirrors `project delete --yes`: destructive intent is opt-in and
explicit, while idempotent re-import stays available (now as an explicit
`--force` operation).

## Scope boundaries

- **`--force` governs *only* the same-name overwrite.** It does **not** relax the
  schema-version or embedding-model compatibility gates — those remain
  unconditional hard refuses (GH-0510). Verified by test.
- **Known limitation, deliberately not expanded here:** a `--force` overwrite
  that fails *mid-landing* leaves the project with the new (registered) DB but
  un-landed `file_path`s, and the prior corpus DB is already replaced — the
  teardown only removes a *freshly-created* project, never a pre-existing one.
  This is the pre-existing re-import-overwrite semantics #526 gates behind a flag,
  not a regression it introduces; making overwrite transactional is out of scope.

Additive, no schema change, `SCHEMA_VERSION` stays 9. Part of omnibus #494.
