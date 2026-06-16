# GH-0633: Self-verifying, shell-safe finding writes

## Context

Two papercuts in the findings write API:

1. `save_finding` and `edit_finding` returned `body`, `chunk_ids`, `citations`, and
   `external_citations` but **not** `title` or `description`. A title-only or
   description-only edit therefore required a follow-up `read_finding` to confirm
   what landed — defeating the self-verifying round-trip the body already provides.

2. `--title` and `--description` on both scripts were bare `type=str` shell args,
   with no file-based alternative. In practice an intended title
   `"…spending ~$8M/quarter in-house"` saved as `"…spending ~M/quarter in-house"`
   because the shell expanded `$8M` to nothing. The body came through `--body-file`
   untouched, which is precisely why that mechanism is correct.

## Decisions

### Echo `title` and `description` in every write response (A.2)

Both `save_finding` and `edit_finding` now include `title` and `description` in
their output JSON, immediately after the provenance fields and before `body`. This
makes every write response a complete snapshot of the finding's metadata — a mangled
title is visible in the same call that wrote it, with no follow-up required.

### Add `--title-file` / `--description-file` (A.3)

Both scripts now offer file-based siblings for the two text args:

- `save_finding`: mutually exclusive groups, each `required=True` — caller must
  pick one of `--title`/`--title-file` and one of `--description`/`--description-file`.
- `edit_finding`: same mutual exclusion groups, no `required` (all three flag pairs
  remain optional; the existing `NOTHING_TO_UPDATE` guard still fires when none
  are passed).

The value is read verbatim via `Path.read_text(encoding="utf-8")` — no shell
expansion, no strip. Error codes are explicit constants (`TITLE_FILE_NOT_FOUND`,
`DESCRIPTION_FILE_NOT_FOUND`) rather than dynamically generated strings, so they
are greppable.

### Shared helper `read_text_arg` in `_common.py`

A single function handles the value-or-file routing for both scripts, keeping the
logic in one place. Signature: `read_text_arg(value, file, *, flag, error_code)`.

### Body-derivation skipped

The issue listed "derive title from body's first H1, description from first
paragraph" as an optional "and/or". It was skipped: the required core (file flags
+ echo) is complete without it, and body-derivation would add complexity that the
smallest-fix rule argues against.
