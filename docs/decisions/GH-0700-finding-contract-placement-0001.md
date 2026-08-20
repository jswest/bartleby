# The tightened `save_finding` contract lives where findings are already discussed in prose, not in a new section — and the two tool-knowledge facts go where their failure mode already gets diagnosed.

Issue #700: an agent saved a session diary entry (a correction narrative for
an in-session mistake that was never itself saved, several unrelated claims
bundled into one record, addressed to itself rather than a future reader) as
a `save_finding` call. The findings UI then presented it with the same
evidentiary chrome as a real finding, and it fed forward into later
sessions' memory search. The issue specified two skill-layer fixes and no
schema/code changes; the judgment calls below are about *where in
`bartleby/skill/SKILL.md`* to land each one, given the standing instruction
to tighten existing prose rather than bolt on a disconnected section.

**Contract tightening → the "you've reached a conclusion worth preserving"
paragraph in `## Output`, not the `save_finding` table row.** The Available
Scripts table row for `save_finding` is already dense with shell-safety and
argument mechanics; it's the wrong place for a five-bullet "what counts as a
finding" contract. The `## Output` section's closing paragraph — "When
you've reached a conclusion worth preserving... call `save_finding`" — is the
one spot in the file that already answers *when/what*, so the new material
(don't staple unrelated records together, write for a zero-context reader,
no correction narratives, no methodology/diary/self-assessment) extends that paragraph in
place rather than opening a new heading.

**FTS5-has-no-stemming → the end of "Zero-result diagnosis," not the `scan`
table row.** The fact belongs wherever an agent is deciding whether a zero
or low count means "absent." The Zero-result diagnosis section already walks
that exact decision (`absent` / `out_of_scope` / `heading_only` /
`filtered_out`) and ends on "the diagnosis is a hint, not a guarantee" —
the stemming caveat is one more way a `scan`/FTS reading can mislead, so it
reads as a continuation of that guidance rather than a new fact bolted onto
the `scan` row's already-long matching-mechanics prose.

**Chunk-boundary negatives → the end of "Citing chunks correctly," not a
new subsection under "Reading search results."** That section already
walks two failure modes for *positive* citations (snippet truncation,
citing an unread chunk) and ends on "if a chunk_id never showed up earlier
as something you read in full, you're guessing — stop and fetch." The
chunk-boundary caveat is the mirror case for *negative* claims ("the
document doesn't say X"), so it reads as a third failure mode in the same
list rather than a standalone note elsewhere.

**Amended on owner review: the unit rule was dropped in favor of naming the
anti-pattern.** The contract as first drafted said "one claim per finding,"
which collides with the guide's own structured-deliverable guidance (a table,
comparison, or timeline *is* a multi-claim body) and with wide-ranging asks —
"five to ten things this corpus says about X" should come back as *one*
finding, not seven. The bullet now states what we don't want — unrelated
records stapled together: separate concerns a future reader would never
search for together — and explicitly lets the shape of the ask set the shape
of the finding, keeping the model maximally responsive to user inputs that
ask it to range widely. The #700 diary entry still fails the contract on
every remaining rule: not about the documents, self-addressed, and a fusion
of separate concerns. (This also moots the earlier question of whether "one
claim per finding" conflicted with the "save interim findings when the work
is long" loop step — there is no unit rule to conflict.)
