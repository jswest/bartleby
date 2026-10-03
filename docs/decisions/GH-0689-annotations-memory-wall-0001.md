# Skill-side finding annotations (annotate and delete) gate on the parent finding through the existing chokepoint; agent notes are `is_human_author=0` and session-stamped; annotations stay out of ranked `search`.

Issue #689 (settled design) and #732 (the skill child) add `annotate_finding`
and annotation read-back to `read_finding` / `list_findings`. Three calls
were made about how annotations meet the memory wall and authorship.

**Annotations have no wall of their own — visibility is the parent
finding's, enforced by `_common.assert_findings_accessible`.** An annotation
anchors to a verbatim quote of the finding body and is read back alongside
it, so writing one discloses and extends the finding. That is exactly the
ARCHITECTURE.md gate rule ("a command gates iff it would disclose, mutate,
or destroy a finding the caller didn't author"), so `annotate_finding`
(`action="annotate"`) gates on the parent finding id before any write,
`delete_annotation` resolves the note's parent via `db.annotations.get_annotation`
and gates on it the same way before deleting or echoing, and
`read_finding` / `list_findings` surface annotations only for findings they
already admit. A second, annotation-specific ownership check was rejected:
it would re-derive the wall per script, the drift GH-0288 consolidated into
one chokepoint. Consequence: in a memory-off session an agent can annotate
its own findings (and see every note on them, including a human's), but
cannot annotate — or learn the annotations of — another session's finding.

**Agent-written annotations are `is_human_author=0` and stamped with the
authoring `session_id`.** The skill never claims human authorship; web and
CLI write `1`. The flag is how a reading agent knows a note outranks the
finding text (SKILL.md: a human note saying an assertion is wrong is carried
into any report), so the skill surface must not be able to forge it — there
is no flag for it.

**Annotations stay out of ranked `search`.** Finding hits are chunk-shaped
fragments; bolting notes onto them (or a `has_annotations` flag) is a
follow-up. `read_finding` and `list_findings` (`annotation_count`) are the
read surfaces that matter: the next session reads the whole finding before
relying on it, and that is where the correction lands.
