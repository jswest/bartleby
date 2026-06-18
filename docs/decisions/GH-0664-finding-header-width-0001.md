# Finding header constrained to the body column; gutter rule padding increased (issue #664)

> Source: [#664](https://github.com/jswest/bartleby/issues/664)

## Change 1 — Finding header width: prose column, not the full gutter-inclusive width

**Decision:** `.ledger-hed`, `.ledger .meta`, and `.ledger-dek` are each constrained
to `max-width: var(--ledger-prose-width)` (currently `36rem`).

**Supersedes GH-0643**, which deliberately set those same selectors to
`max-width: var(--ledger-content-width)` — the prose + gap + notes combined width —
so the header's right edge would align with the far edge of the margin-note gutter.
That alignment was a considered choice at the time but felt out of step with the
prose column that follows it.

**Owner preference (2026-06-18):** the header should align with the body/prose
column, not the margin-note gutter. The prose beneath the header is capped at
`--ledger-prose-width`; the title, meta line, and dek should share that same cap so
the header reads as a true preamble to the prose — not as a wider band that bleeds
into the marginalia region.

**`.toolbar` was not changed.** There is no `.toolbar` rule in the `.ledger`-scoped
section, so no other header element needed attention.

## Change 2 — `.cite-notes` padding: `--space-xl` (up from `--space-lg`)

`.cite-notes { padding-left }` was increased from `var(--space-lg)` to
`var(--space-xl)` to give the margin-note text more breathing room away from the
thin `border-left` rule that separates the marginalia from the prose column.

No structural changes were made to the ledger geometry variables
(`--ledger-prose-width`, `--ledger-notes-width`, `--ledger-content-width`), the
`.ledger-column` grid definition, or the margin-note JS layout logic.
