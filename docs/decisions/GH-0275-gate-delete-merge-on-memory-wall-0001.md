# `delete_finding` and `merge_findings` gate on ownership under memory-off

**Issue:** #275 · **Status:** settled · **Supersedes:** part of GH-0056

## Context

GH-0056 added the curation primitives `delete_finding` and `merge_findings` and
deliberately left both *ungated*: "Neither command gates on `memory_enabled` —
they act on an explicit `finding_id` like the `save`/`edit` write siblings (only
the *reveal* commands `list_findings`/`read_finding` raise `MEMORY_OFF`)." The
#276 memory-wall sweep revisited that call and found the write-sibling analogy
doesn't hold for these two:

- **`delete_finding`** echoes the deleted finding's `title` in its response — a
  content reveal of another session's work — *and* destroys that work outright.
- **`merge_findings`** consumes the `--from` sources (deleting them) and folds
  them into the target, mutating and erasing other sessions' findings.

Both cross the same wall `read_finding` enforces, in the same direction the
#276 sweep was hardening: a `memory_enabled=0` evaluation run must neither read
nor disturb another session's findings. The motivating use case for memory-off
— the multi-model report eval, where each backend's findings are the comparison
dataset — is exactly where letting a run delete or cannibalize a sibling
session's findings is most damaging. (The companion call, GH-0272, gates
`edit_finding` on the same reasoning: gate iff the response discloses or mutates
content the caller didn't supply.)

## Decision

Gate both on ownership, mirroring `read_finding.py:72` — the deliberate choice
between GH-0056's two options ("record + pin the ungated behavior" vs. "gate
like `read_finding`") resolved in favor of gating.

- **`delete_finding`:** a memory-off session deleting a finding authored by
  another session raises `{"code": "MEMORY_OFF"}`, fired *before* the title is
  read back or any row is deleted.
- **`merge_findings`:** under memory-off, *every* finding involved — the `--into`
  target and all `--from` sources — must be self-authored; any foreign finding
  raises `MEMORY_OFF` with a `foreign_finding_ids` list, fired before the
  `with conn:` write block (so a rejected merge deletes nothing).
- A session's **own** findings stay fully curatable under memory-off (a run can
  retract or consolidate its own drafts), and **memory-on** behaviour is
  unchanged.

## Notes

- **No schema change.** A read-time ownership check; no `SCHEMA_VERSION` bump
  (stays 8), no re-ingest.
- **Per-script gate, by design.** Like GH-0271/GH-0272, the ownership check is
  reproduced inline rather than promoted to `_common`, to keep the #276
  sub-issues on mutually-disjoint files. The single-chokepoint consolidation
  (route all five sites — `read_finding`, `read_chunks`, `edit_finding`,
  `delete_finding`, `merge_findings` — through one `_common` predicate) is
  tracked as #288, sequenced to land last on omnibus #276.
- **Smallest fix that fits.** Reuses `read_finding`'s `MEMORY_OFF` code and
  message shape so the agent sees one consistent wall across the read *and*
  curation surfaces.
