# Contributing

Hello! This file is the short story of *how* Bartleby gets built, mostly aimed at
the small circle of people who hack on it — and at anyone curious enough to look
under the hood. It's a description of how we work, not a contract. If something
here is unclear or seems wrong, that's a bug in this doc; say so.

The one-sentence version: **we build Bartleby hand-in-hand with [Claude
Code](https://claude.com/claude-code), and the workflow that makes that pleasant
is version-controlled right in the repo** — so you inherit it for free.

## How we build Bartleby

Open the repo in Claude Code and the project's tooling loads automatically. There's
nothing to install and nothing to copy: the `.claude/` directory ships in the repo,
so a fresh clone already has the skills, the safety hook, and the helper agents we
use day to day. (This is deliberate — see [#130](https://github.com/jswest/bartleby/issues/130)
for when we version-controlled it.)

You don't *have* to use any of this. Plain `git` and a text editor work fine, and
the manual version of the loop is spelled out below. But if you're driving Claude,
the tooling encodes the conventions so you don't have to keep them all in your head.

## What's in `.claude/`

| Piece | What it does |
| --- | --- |
| `skills/ship/SKILL.md` | The `/ship #<N>` command — runs an issue end-to-end into a tested PR (the loop below). |
| `skills/release/SKILL.md` | The `/release` command — dry-run → confirm → publish a release (see [Cutting a release](#cutting-a-release)). |
| `hooks/guard-main-write.sh` | A safety rail that refuses commits/pushes on `main`. |
| `agents/simplify-refactor.md` | A subagent that does a quality/simplification pass over changed code. |
| `agents/git-workflow-manager.md` | A subagent that turns a pile of changes into clean, atomic commits. |
| `settings.json` | Wires the hook in. |

The `.gitignore` is set up so exactly those shared pieces are tracked; the personal
and transient bits (`settings.local.json`, `scratch/`, `worktrees/`, `agent-memory/`)
stay out of version control. So your local experiments don't end up in everyone
else's clone.

## The everyday loop: `/ship #<N>`

Almost all work starts from a GitHub issue and ends as a PR that closes it. Typing
`/ship #141` walks Claude through this, in order:

1. **Sync `main`** and make sure the tree is clean.
2. **Create a sibling worktree** for the issue — `../bartleby-issue-<N>-<slug>` on a
   branch `issue/<N>-<slug>`. We work in worktrees *next to* the main checkout,
   never nested inside it, and never `git checkout -b` on `main` itself.
3. **PAUSE — plan.** For anything non-trivial, Claude lays out the plan (files,
   approach, trade-offs) and waits for your OK before writing code.
4. **Implement in logical units.** For *every* commit, in this exact order:
   `uv run pytest` (must pass) → run the `simplify-refactor` agent over the touched
   files → apply the suggestions worth taking → re-run the tests → commit.
5. **Docs sweep.** Check whether the change needs README / `ARCHITECTURE.md` /
   `SKILL.md` updates (a new flag, changed behavior, a decision-log entry).
6. **Reconcile** with `origin/main` so any merge conflict surfaces now, not in the PR,
   then run the full test suite again.
7. **PAUSE — PR.** Claude shows you the PR body and a final diff summary and waits
   for your OK, then opens a PR with a `Closes #<N>` line.

**Claude opens the PR; a human merges it.** That last step is always ours.

The two **PAUSE** points are real stops — Claude won't blow past them without your
say-so. They're where you catch a wrong approach before it's code, and a wrong PR
before it's public.

### The guard rail

The `guard-main-write.sh` hook blocks `git commit` and `git push` whenever the
branch is `main`. This is **intentional, not a bug.** If you hit
`BLOCKED: refusing 'git commit'…`, it means you're on the wrong branch — switch to
your issue's worktree branch and try again. (Step 2 above keeps you out of this;
the hook is the backstop for when something slips.)

### `with-playwright` (web changes only)

For changes under `bartleby/web/`, you can append a `with-playwright` token —
`/ship #<N> with-playwright` — to turn on a visual-verification loop: Claude drives
a real browser, screenshots the affected routes before and after, and iterates on
what looks wrong. It's **off by default** because browser automation is slow and
burns image tokens, so you opt in per run when a change is actually worth eyeballing.
Backend-only issues ignore the token even if you pass it.

### Skipping the test gates (docs-only changes)

The `uv run pytest` runs that otherwise gate every commit are skipped down two
paths — both a convenience for diffs that can't affect tests, **not** a way to land
untested code:

- **Docs-only — automatic, no token.** When every changed path is a `*.md` file,
  `LICENSE`, or under `docs/` — README wording, an `ARCHITECTURE.md` note, a
  `SKILL.md` tweak — Claude skips the gates on its own. A pure prose change can't
  move the suite, so there's nothing to gate and no token to remember.
- **`skip-tests` — opt-in, for docs-adjacent files outside that set.** For a change
  that's still test-irrelevant but touches something other than docs (a shell
  script, a `.txt` asset), append a `skip-tests` token (`/ship #<N> skip-tests`).
  Claude honors it only when the branch diff touches no `*.py`, `pyproject.toml`, or
  `bartleby/web/` file — otherwise it runs the tests anyway and tells you why.

Either way, Claude re-checks at every gate, so a docs PR that grows a code change
mid-stream starts running tests from that point. When tests are genuinely skipped,
the PR and final report say so — naming which path applied. Combine `skip-tests`
with `with-playwright` in either order.

### `onto #<omnibus>` (ship onto an omnibus branch)

For a bundle of related issues that should reach `main` as one unit — a release,
say — you can stage the sub-issues on a shared *omnibus branch* first. Append an
`onto #<omnibus>` token, where `#<omnibus>` is the **issue number** tracking the
bundle: `/ship #170 onto #169`. Claude reads that omnibus issue, derives its branch
from the title's leading version (`v0.8.0 — …` → `omnibus/v0.8.0`), and creates the
branch off `main` on the bundle's *first* ship — asking first, since that pushes a
new long-lived branch. From there the whole loop retargets from `main` to the
omnibus branch: worktree base, collision scan, reconcile, and PR base. The sub-PR
says *"Part of #169"* rather than `Closes #170`, because GitHub only auto-closes
from the default branch; the sub-issues all close when the omnibus → main PR (which
lists every `Closes #<N>`) finally merges. The `main`-only guard rail is unchanged,
so the omnibus branch itself isn't hook-protected — keeping work on sub-PRs is
discipline, not enforcement. Composes with the two tokens above.

### The helper agents

Two subagents do focused jobs so the main thread stays on the problem:

- **`simplify-refactor`** — a quality pass: hunts duplication, needless abstraction,
  and bloat in the code you just touched. It's about clarity, not correctness; it's
  part of every commit's gate above.
- **`git-workflow-manager`** — groups changes into small, clearly-messaged,
  single-concern commits when a change has gotten tangled.

## Working agreements

The full set of invariants and the decision log live in
[`ARCHITECTURE.md`](./ARCHITECTURE.md) — read that before changing anything
load-bearing. The two that shape day-to-day work most:

- **No backwards compatibility, by default.** We delete old code rather than leaving
  dormant compat shims or feature-flagged old paths. The one sanctioned exception is
  *additive-only* schema upgrades (new tables, indexes, nullable columns), which ship
  with an entry in `bartleby/db/upgrades.py` so existing corpora can `bartleby project
  upgrade` instead of re-ingesting.
- **Schema bumps mean re-ingest.** A non-additive schema change bumps `SCHEMA_VERSION`
  and tells users to re-ingest; there's no automatic migration for those. (This is also
  why the release version's *minor* number is the schema version — see the README.)

## Running the tests

```
uv run pytest
```

That's the whole suite. The per-commit gate above runs it before and after every
change; run it yourself anytime.

## Cutting a release

Releases are a deliberate, batched act on `main` after a few changes have landed —
not something done per-PR. Typing `/release` walks Claude through it: enforce the
on-`main` / clean / synced preconditions, dry-run [`scripts/release.py`](./scripts/release.py)
to show the computed next version and notes, **PAUSE for your explicit OK**, then
`--tag --push` to publish and report the release URL. The script stays the source
of truth for the version math and the schema-drift guard; the skill only advises
and orchestrates (and never invents a version or reaches around the guard hook).
The consumer side — pinning to and upgrading between releases — is documented in
the README under [Pinning to a release](./README.md#pinning-to-a-release) and
[Upgrading from a release](./README.md#upgrading-from-a-release).

## A note for non-Claude contributors

None of this requires Claude. The loop is just good hygiene: branch in a worktree,
keep commits atomic and tested, run `uv run pytest` before you commit, open a PR that
closes the issue, and let someone else merge it. `/ship` automates the bookkeeping;
it doesn't replace your judgment. The machine-readable companion to this file is
[`CLAUDE.md`](./CLAUDE.md) — the same conventions in terse, imperative form, which
Claude auto-loads when you open the repo.
