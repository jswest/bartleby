# Omnibus issues carry a comment-anchored checkbox block that `/ship` reads and writes (issue #297)

> Source: [#297](https://github.com/jswest/bartleby/issues/297)

The `ship` skill's `onto #<omnibus>` flow already *wanted* to keep an omnibus
issue's tracking current: on sub-PR-open it ticks "this issue's box `[ ]→[x]`",
and promotion mode (`/ship #<omnibus>`) builds its `Closes #<N>` manifest "from
the omnibus issue's hand-curated sub-issue checklist". Both steps assumed a
machine-anchorable per-sub-issue checkbox existed. None did. Real omnibus bodies
used free-prose bullets — #282 listed sub-issues as `- #253 — …` and even bare
`- #294` / `- #295` lines, #117 as `- #48 — …` — so there was no `[ ]` to flip
and no structured `#<N>` set to read. The tick and manifest steps degraded to
"report it and leave the body untouched", which was the *common* case, not the
exception.

This defines one regime instead of leaving the format implicit: omnibus bodies
carry a **comment-delimited checklist block**

```
<!-- omnibus-checklist:start -->
- [ ] #<N> — <short label>
<!-- omnibus-checklist:end -->
```

one `- [ ] #<N>` line per sub-issue, kept *alongside* the prose `### Sub-issues`
narrative rather than replacing it. `/ship` (a) **seeds** the block — proposing
it at the step-11 PAUSE, never silently — when the bundle's first ship finds the
omnibus body has none; (b) **ticks** by matching a line's `#<N>` and flipping its
box, annotating with the sub-PR number; (c) **reads** the block's `#<N>` set as
the promotion `Closes` manifest, still reconciled against actually-merged
sub-PRs. A body that predates the regime degrades gracefully: the tick step
proposes adding the line/seeding the block at the PAUSE (declined → report and
leave untouched), and the manifest falls back to the prose list.

**Why comment anchors, not native task lists.** GitHub's `- [ ] #N` issue-
reference task lists would auto-track, but require restructuring the omnibus body
away from the human-readable prose checklist — the same rejection [#265](https://github.com/jswest/bartleby/issues/265)
made. Comment markers keep the prose narrative and give `/ship` a stable,
parseable, anchored region to edit without free-form rewriting a hand-curated
body.

**Relationship to #265 and #191.** This is the *format/discipline* layer; #265
is the *concurrency-safe writer* over it — a single serialized reconciler that
regenerates the anchored block, so parallel sub-ships can't clobber each other's
flips. #265 needs exactly this well-delimited block to regenerate; #297 defines
and bootstraps it. #191 wants to codify the post-merge anchored-edit habit in
SKILL.md; a consistent block is what makes that habit reliable rather than
conditional. Skill/docs-only change — no `SCHEMA_VERSION` bump, no code path.
