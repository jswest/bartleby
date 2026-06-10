# Schema-change versioning policy — minor = `SCHEMA_VERSION`, additive-vs-breaking is binary (issue #362)

> Source: [#362](https://github.com/jswest/bartleby/issues/362)

The convention behind the version number lived in someone's head, not in writing —
the "held at 8" episode (the concurrent-ingestion omnibus shipping *staying at*
schema v8, see [GH-0212](GH-0212-complete-v7-to-v8-additive-upgrade-0001.md)) and
the additive-vs-breaking call being re-derived at every bump were both symptoms of
an unwritten rule. This entry writes it down; the mechanics it describes already
exist in the code, so nothing here is new behavior.

The policy, stated once:

- **Any change that touches the schema bumps the minor version.** Releases are
  `v0.<SCHEMA_VERSION>.<patch>` — the minor *is* `SCHEMA_VERSION`
  (`bartleby/db/schema.py`), not merely bumped alongside it
  ([GH-0100](GH-0100-semi-automated-releases-tags-version-pins-minor-0001.md)). A
  DDL change without a matching `SCHEMA_VERSION` bump is the one mistake the version
  arithmetic can't see, so `check_drift` in `scripts/release.py` AST-diffs the
  schema DDL against the last tag and **refuses to tag** when the DDL moved but the
  constant didn't. That refusal is the enforcement mechanism — the policy isn't
  advisory.

- **The disposition of a bump is binary: additive or breaking.**
  - **Additive** — purely new tables, new indexes, or new nullable columns (NULL a
    truthful "not-yet-known", no row transformations, no column renames, no
    semantic shift in existing data). An additive bump **ships a
    `_upgrade_vN_to_vN+1` entry** in the upgrade chain (`bartleby/db/upgrades.py`,
    registered in `_UPGRADES`); users run `bartleby project upgrade <name>` to apply
    it in place. No re-ingest.
  - **Breaking** — anything else (non-additive DDL, or a DDL-additive change that
    needs data populated at ingest — see
    [the additive-upgrade test](additive-upgrade-test.md)). A breaking bump ships
    **no chain entry** for that step, so `project upgrade` refuses and the only path
    forward is **re-ingest**.

  There is no third option and no partial migration: every bump is
  additive-with-an-upgrade-function or breaking-with-re-ingest. The codebase never
  branches on schema version; it pins to `SCHEMA_VERSION` exactly, and the upgrade
  is a one-shot at the gate.

- **The `breaking-schema` label is reserved for re-ingest-required changes.** An
  issue carrying `breaking-schema` is one whose bump strands existing corpora until
  they re-ingest; the release notes for such a release call out the re-ingest
  explicitly. An **additive** bump is *not* labelled `breaking-schema` — its release
  messaging tells users to run `bartleby project upgrade <name>`, not to re-ingest.

The half of this policy that surfaces the distinction *to users* is the release
tooling: [#361](https://github.com/jswest/bartleby/issues/361) makes
`scripts/release.py` additive-aware so the release notes and dry-run say
"run `bartleby project upgrade <name>`" when the crossed versions are covered by a
chain entry and only print the re-ingest banner for a genuinely non-additive bump —
so an additive bump (like the v0.9.0 omnibus this entry ships with) can't
mis-publish a re-ingest order it doesn't need. `check_drift` (this entry) is the
*gate*; #361 is the *messaging* — distinct concerns over the same `release.py`.
