# `edit_finding` gates on ownership under memory-off — refines GH-0056

Issue #272 (part of the v0.8.7 omnibus, #276). Refines
[GH-0056](GH-0056-findings-curation-delete-finding-merge-findings-0001.md),
which grouped `edit_finding` with the *ungated* write siblings: "only the
*reveal* commands `list_findings`/`read_finding` raise `MEMORY_OFF`."

That grouping was wrong for `edit_finding`. Its response echoes the finding's
full `body` and `citations` (the verbatim-echo contract it shares with
`save_finding`), so a `--title`-only edit on a guessable foreign finding —
which keeps the stored body untouched and returns it whole — is a **read-by-write
bypass** of the same wall `read_finding` enforces, with a foreign-row mutation as
a side effect. `edit_finding` is a write command, but it is *also* a reveal
command; the reveal half is what the wall cares about.

## Decision

`edit_finding` raises `MEMORY_OFF` when the session has `memory_enabled = 0` and
the finding was authored by another session — the predicate is byte-identical to
`read_finding.py`'s. The check sits immediately after the `FINDING_NOT_FOUND`
fetch and **before** any validation or write, so a blocked call leaks no body and
mutates nothing. A memory-off session can still edit findings it authored itself.

The line that distinguishes gated from ungated, going forward: **a mutator gates
under memory-off iff its response (or error) discloses content the caller didn't
supply.** `edit_finding` echoes the stored body → gated. `delete_finding` and
`merge_findings` (still ungated, GH-0056) act on an explicit `finding_id` and
return no foreign body — they remain open write siblings (#275 pins that
deliberately).

## Scope

Additive, **non-schema** — a per-script gate mirroring `read_finding`, no shared
helper (the omnibus chose per-script gates over a `_common` ownership predicate
to keep its sibling issues conflict-free). No `SCHEMA_VERSION` bump, no
re-ingest. Touches `bartleby/skill_scripts/edit_finding.py`,
`tests/test_skill_edit_finding.py`, and the memory-off invariant note in
`ARCHITECTURE.md`.
