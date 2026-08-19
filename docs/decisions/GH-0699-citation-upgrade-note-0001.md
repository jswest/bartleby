# A README "After updating Bartleby" callout, not a new doc, tells users to check pre-`[^chunk:N]` findings (issue #699)

> Source: [#699](https://github.com/jswest/bartleby/issues/699)

## The problem

Findings authored before the `[^chunk:N]`-typed citation format landed
([#624](https://github.com/jswest/bartleby/issues/624)) keep whatever
untyped/old-style marker they had. That doesn't error on read — it just
silently stops being recognized as a citation, so the finding's
`finding_citations` rows go stale with no signal. `edit_finding` with a
rewritten body is the working fix (it re-extracts citations from the body and
rebuilds `finding_citations`), but nothing in the product tells a user this is
necessary.

This repo has no `CHANGELOG.md` — release notes live in GitHub Releases,
auto-generated from `git log`
([GH-0100](./GH-0100-semi-automated-releases-tags-version-pins-minor-0001.md)).
So a durable, discoverable callout needs a different home.

## Decision — extend the existing "After updating Bartleby" section

`README.md`'s "After updating Bartleby" section is already the established
convention for "what to check after you pull/update the tool" — it covers the
schema-mismatch case (`bartleby project upgrade <name>` vs. re-ingest)
immediately above. The citation-staleness issue is the same shape of problem
(an update can leave existing data behind) even though it isn't a schema
change, so a new paragraph was added directly after the schema-upgrade
paragraph, before the "Gotchas" section, rather than inventing a new doc.

The callout explicitly notes that a one-time backfill
([#642](./GH-0642-backfill-legacy-citation-markers-0001.md)) already fixed
every corpus present on a given machine when it ran, so this note is really
for two remaining cases: a corpus adopted from elsewhere (corpus-share,
per [no-natural-key-citation-resolver](../../ARCHITECTURE.md)), or one that
predates the #642 fix on its machine. Omitting that context would have made
the README claim *all* old findings are currently stale, which is false for
already-backfilled corpora and would be needless alarm.

## Why not `bartleby/skill/SKILL.md` or `README.md`

The skill docs already describe `edit_finding` as the tool to "fix malformed
citations or revise" — that's the *mechanism*, aimed at an agent mid-session.
This callout is aimed at a *human* deciding whether to go check their old
findings after updating the tool, which is a README-shaped concern (install/
upgrade guidance), not a skill-contract concern. No change was made to the
skill docs.
