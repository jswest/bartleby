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
| `skills/ship/SKILL.md` | The `/ship #<N>` command — runs an issue end-to-end into a tested PR, or fans an omnibus out to players (the loop below). **Vendored** (see note). |
| `skills/release/SKILL.md` | The `/release` command — dry-run → confirm → publish a release (see [Cutting a release](#cutting-a-release)). Repo-local (not vendored). |

> **Vendored skill.** `ship` is *pressed* verbatim from a local skill drawer (one
> source, many repos) by `signet`, so don't hand-edit its `SKILL.md` here — edit
> the drawer and re-press. Everything bartleby-specific lives in `.claude/ship.toml`
> (committed; the press never touches it). `tests/test_skill_drift.py` fails if the
> pressed copy drifts from the drawer, and skips when the drawer is absent (a clone /
> CI). `release` stays repo-local.
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
`/ship #141` kicks off a three-role workflow:

**Roles:**
- **Director** — the session itself (you, running Claude Code). Plans, dispatches
  players, triages critic findings, assembles omnibus branches, opens promotion PRs.
  Runs at the strong tier.
- **Players** — subagents that implement one issue each, each in its own sibling
  worktree. Run at the cheap tier by default (token spend concentrates here).
- **Critics** — a correctness critic (bugs, integration seams) and a simplicity
  critic (duplication, needless abstraction), spooled by the director. Advisory only;
  the director triages their findings.

**Leaf vs. omnibus:** the fork is a single API check — an issue with **native GitHub
sub-issues** is an omnibus; anything else is a leaf. Checklists and title
conventions do **not** decide it.

For a **leaf issue**, the director:

1. Creates a sibling worktree off `main` — `../bartleby-issue-<N>-<slug>`, never
   nested inside the checkout, never `git checkout -b` on `main`.
2. Dispatches one player with the issue spec, guardrails, and tier.
3. The player implements, runs `uv run pytest`, self-tidies, commits, and opens the
   **issue → `main` PR**.
4. The correctness and simplicity critics review the branch; the director triages.
5. **Docs sweep** — README / `ARCHITECTURE.md` / `SKILL.md` brought in line.
6. **PAUSE — PR.** Director shows you the PR body and waits for your OK. Claude
   opens the PR with a `Closes #<N>` line. **You merge.**

**Claude opens the PR; a human merges it.** That step is always yours. After you
merge, the director removes the sibling worktree and re-syncs the base.

**PAUSE** points are real stops — Claude won't blow past them. They're where you
catch a wrong approach before it's code, and a wrong PR before it's public.

### Plan gate

Before any risky or underspecified work — and *especially* before an omnibus
fan-out — the director presents the plan (or the wave DAG for an omnibus) and waits
for your OK. Trivial, unambiguous leaves may skip the gate; everything uncertain
stops. `--plan` makes it mandatory.

### The guard rail

The `guard-main-write.sh` hook blocks `git commit` and `git push` whenever the
branch is `main`. This is **intentional, not a bug.** If you hit
`BLOCKED: refusing 'git commit'…`, it means you're on the wrong branch — switch to
your issue's worktree branch and try again. (Step 2 above keeps you out of this;
the hook is the backstop for when something slips.)

### Optional flags

- **`--with-playwright`** — for changes under `bartleby/web/`, turns on a
  visual-verification loop: a player drives a real browser, screenshots affected
  routes before and after, and iterates on what looks wrong. Off by default (slow;
  burns image tokens). Available only when `.claude/ship.toml` declares a
  `[playwright]` block. Ignored for backend-only issues.
- **`--strong`** / **`--thrifty`** — tier overrides. `--strong` lifts every role to
  strong; `--thrifty` drops all to cheap. Mutually exclusive.
- **`--plan`** — force the plan gate before any work, even for trivial leaves.
- **`--skip-tests`** — see [Skipping the test gates](#skipping-the-test-gates).

### Skipping the test gates

The `uv run pytest` runs that otherwise gate every commit are skipped down two
paths — both a convenience for diffs that can't affect tests, **not** a way to land
untested code:

- **Docs-only — automatic, no token.** When every changed path is a `*.md` file,
  `LICENSE`, or under `docs/` — README wording, an `ARCHITECTURE.md` note, a
  `SKILL.md` tweak — Claude skips the gates on its own. A pure prose change can't
  move the suite, so there's nothing to gate and no token to remember.
- **`skip-tests` — opt-in, for test-irrelevant files outside that set.** For a change
  that can't affect the suite but touches something other than docs — a frontend-only
  edit under `bartleby/web/` (that tree is all Svelte/vite, no Python), a shell
  script, a `.txt` asset — append a `skip-tests` token (`/ship #<N> skip-tests`).
  Claude honors it only when the branch diff touches no `*.py` or `pyproject.toml`
  file — otherwise it runs the tests anyway and tells you why. (One caveat: a
  *structural* `bartleby/web/` change (moving `src/`, dropping `package.json`) can
  still break the Python suite via `tests/test_serve.py`, which checks the packaged
  UI layout — don't `skip-tests` a web restructure.)

Either way, Claude re-checks at every gate, so a docs PR that grows a code change
mid-stream starts running tests from that point. When tests are genuinely skipped,
the PR and final report say so — naming which path applied. Combine `skip-tests`
with `with-playwright` in either order.

### `onto #<omnibus>` (ship a leaf onto an omnibus branch)

`--onto <branch>` (or its shorthand `onto #<omnibus>`) lands a single leaf onto an
existing omnibus branch — adding to an in-flight omnibus by hand. The sub-PR says
*"Part of #<omnibus>"* rather than `Closes`, because GitHub only auto-closes from
the default branch; the sub-issues close when the omnibus → main PR merges.

### Omnibus: `/ship #<N>` on a parent issue

When the argument issue has **native GitHub sub-issues**, `/ship` takes the omnibus
path automatically. The director:

1. Ensures the **omnibus branch** exists (creates it off `main` if missing, asking
   first — that's a remote mutation).
2. Computes a **wave plan** from the sub-issues (topological by declared
   dependencies, then by file-overlap), posts it as an omnibus comment, seeds the
   progress checklist (`<!-- omnibus-checklist:start/end -->`), and **pauses for
   your OK** before fanning out.
3. Dispatches sub-issues to **players in parallel** (one per sub-issue, each in its
   own sibling worktree off the current omnibus HEAD).
4. Runs a **merge-train** — per sub-PR: simplicity critic reviews the diff, small
   fixes applied, then integrated and tested on the omnibus branch; conflict or red →
   **park** (never resolved unattended); checklist line ticked on landing.
5. After all waves land: correctness critic and cross-cutting simplicity critic
   review the assembled omnibus; triage loop up to `max_critic_passes`.
6. Docs sweep; then **promotion PR** (omnibus → `main`) with `Closes #<N>` for
   every sub-issue. **You merge** — that one merge closes the whole bundle.

One authority rule applies here (see
[`docs/decisions/GH-0244-…`](./docs/decisions/GH-0244-ultraship-authority-boundary-0001.md),
refined by [`GH-0335-…`](./docs/decisions/GH-0335-ultraship-subagent-players-gate-at-stage-manager-0001.md)):
the director merges sub-issue → omnibus autonomously (recoverable branch, serialized,
gated, never force-pushed). Players are **subagents** under the session's own
permission posture — no pre-granted permissions, no 1 a.m. prompts. The boundary
that does **not** move: **a human merges omnibus → `main`**, and `guard-main-write.sh`
is untouched.

The state is reconstructable from GitHub at any restart (merged sub-PR = done, open =
parked, neither = untouched), so an interrupted run can resume without replaying a
journal. For a large omnibus you want to run while you're away, combine `/ship` with
a loop or scheduled agent — the director handles everything; the omnibus path is fully
unattended except for the plan gate and the final merge.

### Configuring `/ship`: `.claude/ship.toml`

Repo-specific facts live in `.claude/ship.toml` (committed; `signet` never touches
it). Key fields: `base_branch`, `test_cmd` / `test_source_globs` / `docs_only_globs`,
`gate_agent` (the per-commit quality pass), and the `live_data_note` redline (injected
verbatim into the implementation guidance — share it in git so a fresh clone gets the
same redline). On first run with no `.toml`, `/ship` auto-detects and proposes a
config; `/ship config` or `/ship --reconfigure` re-runs it on demand.

`/ship` also runs a **signet self-check** on startup: it reads
`.claude/skills/ship/.stamp`, finds the drawer, and byte-compares the pressed files.
Drift → stop and prompt to re-press. Drawer absent (clone, CI) → skip silently.
`--no-signet-check` bypasses it.

### The helper agents

Two subagents do focused jobs so the main thread stays on the problem:

- **`simplify-refactor`** — a quality pass: hunts duplication, needless abstraction,
  and bloat in the code you just touched. It's about clarity, not correctness; it's
  part of every commit's gate above.
- **`git-workflow-manager`** — groups changes into small, clearly-messaged,
  single-concern commits when a change has gotten tangled.

## Working agreements

The full set of invariants lives in
[`ARCHITECTURE.md`](./ARCHITECTURE.md) — read that before changing anything
load-bearing; the decision log behind past calls lives one-per-file under
[`docs/decisions/`](./docs/decisions/). The two invariants that shape day-to-day
work most:

- **No backwards compatibility, by default.** We delete old code rather than leaving
  dormant compat shims or feature-flagged old paths. The one sanctioned exception is
  *additive-only* schema upgrades (new tables, indexes, nullable columns), which ship
  with an entry in `bartleby/db/upgrades.py` so existing corpora can `bartleby project
  upgrade` instead of re-ingesting.
- **Schema bumps mean re-ingest.** A non-additive schema change bumps `SCHEMA_VERSION`
  and tells users to re-ingest; there's no automatic migration for those. (This is also
  why the release version's *minor* number is the schema version — see the README.)
- **The disposition is binary, and the label is reserved.** Every schema change bumps
  the minor (= `SCHEMA_VERSION`; `check_drift` in `scripts/release.py` refuses to tag a
  DDL change that forgot the bump) and is either *additive* (ships a chain entry → users
  run `bartleby project upgrade`) or *breaking* (no chain entry → re-ingest). The
  `breaking-schema` label is reserved for the re-ingest-required case; additive bumps
  aren't labelled breaking. The full policy is
  [the schema-change versioning-policy decision](./docs/decisions/GH-0362-schema-change-versioning-policy-0001.md).

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
