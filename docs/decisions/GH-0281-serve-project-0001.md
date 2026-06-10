# GH-0281 — `bartleby serve` takes an optional `--project`, overriding the active project for that server only

Issue #281 asked us to decide deliberately: should `serve` accept a `--project`
like every other project-scoped command (`scribe`, `session`, `logs`, the skill
runner, `project info/upgrade`), or is its active-project-only design intentional?

**Decision: add the flag.** `serve` accepts an optional `--project <name>`. It is
a cross-surface parity gap — `serve` was the lone project-scoped command that took
no arguments, so browsing a second corpus in the web UI forced a
`bartleby project use <name>`, mutating the *persisted* active project as a side
effect of "just wanting to look." That global mutation is the real cost the flag
removes.

**Override semantics: this server only, never persisted.** `--project` does **not**
write `config.yaml`. The mechanism is one new environment variable,
`BARTLEBY_PROJECT`:

- `commands/serve.py` validates the named project's DB exists (a clean CLI error
  via `project_db_path`, rather than letting the node side throw later), then
  exports `BARTLEBY_PROJECT` into the environment that `os.execvp` inherits.
- `web/src/lib/server/db.js`'s `activeProject()` prefers `process.env.BARTLEBY_PROJECT`
  over `config.yaml`'s `active_project`.

That single seam covers both consumers: the direct read-only DB open *and* the
skill subprocesses (`skill.js` derives its `--project` from `getDb()`), so the web
views and the skills it shells out to all follow the same overridden project.
Without the flag the server falls through to the persisted active project exactly
as before — no behavior change.

**Why not record active-only as intentional (the alternative):** there was no
design reason for the asymmetry; `serve` simply predated the parity. Keeping it
active-only would have left the only way to browse a second corpus as a global
state mutation, which is precisely the friction the issue flagged.

No schema change, no `SCHEMA_VERSION` bump, no re-ingest. Part of the v0.8.9
web-UI / document-presentation omnibus (#282).
