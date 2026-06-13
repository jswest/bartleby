# `probe_dates` skill validates a date regex before the agent prompts a human (issue #538)

> Source: [#538](https://github.com/jswest/bartleby/issues/538)

`scribe backfill-dates` (#536) is a **human-run** bulk write that dates an
undated corpus from a filename regex — but it's deliberately off the agent
surface. An agent facing a temporal task on an undated corpus therefore has to
*tell* a human to run it, and prompting blindly is a wasted round-trip: the date
may not sit in the filename in any regexable form, or a candidate regex may match
poorly. The agent needs a way to **confirm a regex would work** before it says
anything.

**A read-only skill script, the agent-side twin of backfill's `--dry-run`.**
`bartleby skill probe_dates --regex '(?P<date>…)' [--field filename|body]
[--sample N] [--file-like PATTERN]` lives at
`bartleby/skill_scripts/probe_dates.py` and is dispatchable on drop-in (the
dispatcher derives its allowlist from the package directory — no registry edit).
It **writes nothing**: it reports how well the regex extracts dates over the
filenames (default) or a body sample, so the agent can iterate on the regex and
decide whether to prompt. The backfill command's own `--dry-run` stays the
human-side preview of the actual write; same helpers, different actors.

**Reuses #536's helpers, no reimplementation.** Extraction and date-validity go
through `compile_filename_date_regex` (which enforces the named `date` group),
`extract_filename_date`, and `normalize_authored_date` from
`bartleby/ingest/summarize.py`. So a clean probe here predicts a clean backfill —
the two surfaces can't drift in what they consider a match or a valid date. A
malformed regex or a missing `date` group is a usage error surfaced as
`INVALID_REGEX` (JSON envelope, non-zero exit), reusing `FilenameDateError`.

**Output contract.** `sampled` / `sample_size` / `population` / `matched` /
`match_rate` / `normalized_ok` / `normalized_invalid` / `examples` (a few
`file_name → extracted → normalized` good hits) / `unmatched_examples` (a few
file_names the regex missed, to refine against) / `suggested_command` /
`match_threshold`. A match whose captured text isn't a real calendar date
(`2024-13-40`) lands in `normalized_invalid` and **never** in `normalized_ok` —
the same honest-NULL posture backfill takes.

**`suggested_command` is gated on a high match rate (`MATCH_THRESHOLD = 0.9`).**
It is emitted *only* when `match_rate >= 0.9`, carrying the exact
`bartleby scribe backfill-dates <project> --from-filename '<regex>'` line for the
human (regex `shlex.quote`d). 0.9 is deliberately strict: handing a human a
command that dates only a fraction of the corpus invites a wasted round-trip. The
threshold is a published output field (`match_threshold`) so the agent isn't
guessing the cutoff. `match_rate` counts *regex hits* (not normalizable dates),
so a regex that matches everything but captures junk still clears the rate — the
agent reads `normalized_invalid` to see that and refine; invalidity is reported,
not suppressed.

**`--field body` is always sampled.** A full-corpus body scan is expensive, so
`--field body` always sets `sampled: true` and caps at `--sample` (default 200),
matching the body chunks via `GROUP_CONCAT` per document. `--field filename` is
cheap and probes the full corpus unless `--sample` actually bites (population
reaches the cap), in which case `sampled` flips true. `--file-like` scopes the
probe with the same OR'd SQL `LIKE` semantics as `scan` / `search` /
`list_documents`.

**SKILL.md — verify-then-prompt.** A new subsection under the authored-date prose
spells out the flow: on a high `undated_document_count` from `describe_corpus`,
draft a regex, run `probe_dates`, iterate against `unmatched_examples`, and prompt
the human **only** when a `suggested_command` comes back. It states plainly that
the agent **cannot run the backfill itself** — `scribe backfill-dates` is a
human-run admin command, not a skill script. If no regex clears the bar, the
agent reports dates aren't recoverable and proceeds *without* temporal filtering
rather than sending the human on a useless fix; `save_date` stays the
per-document path when a date is actually read in a document's text.

**Out of scope:** non-date facet probing (future, with the #48 sidecar / facets
work); any write path.

Touches: new `bartleby/skill_scripts/probe_dates.py`; new
`tests/test_skill_probe_dates.py`; `bartleby/skill/SKILL.md` (a quick-reference
example line + the verify-then-prompt subsection).
