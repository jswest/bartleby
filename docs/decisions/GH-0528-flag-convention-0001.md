# A consistent `--yes` vs `--force` flag convention across bartleby (issue #528)

`project import --force` (GH-0526) sharpened a latent inconsistency: bartleby was
spelling "I accept the destructive consequence" two different ways (`project
delete --yes`, `project import --force`) while `--force` *also* meant three
unrelated non-destructive things (`ready` reinstall, `read_document` past-budget,
`tag` re-sweep). A caller — human or agent — couldn't predict which flag a command
wanted, or what `--force` would do.

## The convention

Two senses, two flags, never crossed:

- **`--yes` ⇒ "yes, proceed with the irreversible / data-losing action."** The
  destructive-confirmation sense, matching the `apt -y` idiom. Covers `project
  delete` (already) and `project import` overwrite (the one rename — see below).
  These commands prompt interactively unless `--yes` is passed.
- **`--force` ⇒ "override a guard or skip that is *not* itself data loss."** The
  do-it-anyway sense. Covers `ready` (reinstall an already-up-to-date skill),
  `read_document` (read past the `max_read_tokens` budget guard), and `tag`
  (re-classify documents that already carry assignments). All three are
  **unchanged**.

A new flag picks its name by this split, not by habit. The standing rule lives in
`ARCHITECTURE.md` (Conventions); this file is the *why*.

## What actually changed

One rename: **`project import --force` → `--yes`**, because overwriting a
same-named project drops its local findings — a data-losing act, so it belongs to
the `--yes` family. The three non-destructive `--force`s keep their name, which
means **no agent-facing skill-script flag changed** (`read_document`/`tag` still
take `--force`); `SKILL.md` needed no edit, the safest possible outcome for the
harness contract.

## Short forms

**None.** No `-y`, no `-f`, anywhere. `project delete` previously carried a `-y`
short form (the only command that did); it is dropped so the surface is uniform —
long forms only. Uniformity beats the one saved keystroke, and an unmemorable
half-set of short forms is worse than none.

## Interactive parity for `import` — amends GH-0526

`project import` now **prompts before overwriting** (a `Confirm.ask` in the CLI
command handler) unless `--yes` is given, exactly like `project delete`.

GH-0526 had *rejected* an interactive confirm for import as "hostile to unattended
import." That objection was to a confirm offered **instead of** a flag — which
would force a human at the keyboard for every scripted overwrite. This decision
amends that one point: the prompt is **additive to** `--yes`. An unattended caller
passes `--yes` and never reaches the prompt — the same one-flag cost as the old
`--force` — so the no-silent-overwrite contract GH-0526 established is fully
preserved while interactive use gets a friendlier confirm instead of a bare error.
The rest of GH-0526 stands: overwrite is opt-in, decided up front before any
download, and never relaxes the schema/embedding compatibility gates.

The library `import_project(..., force=...)` keeps its `force` parameter — it is
the primitive "actually overwrite" boolean, and direct (non-CLI) callers still use
it. Only the user-facing CLI flag and its handler renamed to `yes`; the handler
translates a confirmed prompt or `--yes` into `force=True`.

## Scope

Additive/rename-only — **no schema change**, `SCHEMA_VERSION` unchanged. Renamed
flags are clean breaks per CLAUDE.md: no hidden `--force` alias on `import`, no
`-y` alias on `delete`.
