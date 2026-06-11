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
| `skills/ultraship/SKILL.md` | The `/ultraship` command — assembles a whole omnibus bundle unattended (see [Assembling a bundle unattended](#assembling-a-bundle-unattended-ultraship)). Backed by `scripts/ultraship.py`. |
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

1. **Sync `main`, check the tree is clean, and scan for collisions** — a quick look
   at the other open worktrees and PRs so you can spot overlapping in-flight work on
   the same files and coordinate before two branches diverge.
2. **Create a sibling worktree** for the issue — `../bartleby-issue-<N>-<slug>` on a
   branch `issue/<N>-<slug>`. We work in worktrees *next to* the main checkout,
   never nested inside it, and never `git checkout -b` on `main` itself.
3. **Flesh out a thin issue.** If the body is empty or sketchy, Claude writes a
   problem/approach/scope back to it with `gh issue edit` before touching code — so
   the plan has something concrete to anchor to.
4. **PAUSE — plan.** For anything non-trivial, Claude lays out the plan (files,
   approach, trade-offs) and waits for your OK before writing code.
5. **Implement in logical units.** For *every* commit, in this exact order:
   `uv run pytest` (must pass — except down the two [skip paths](#skipping-the-test-gates)
   below) → run the `simplify-refactor` agent over the touched files → apply the
   suggestions worth taking → re-run the tests → commit.
6. **Docs sweep.** Check whether the change needs README / `ARCHITECTURE.md` /
   `SKILL.md` updates (a new flag, changed behavior, a new `docs/decisions/` entry).
7. **Reconcile** with `origin/main` so any merge conflict surfaces now, not in the PR,
   then run the full test suite again.
8. **PAUSE — PR.** Claude shows you the PR body and a final diff summary and waits
   for your OK, then opens a PR with a `Closes #<N>` line.

**Claude opens the PR; a human merges it.** That last step is always ours. Once
you've merged, Claude closes the loop — removing the sibling worktree, deleting the
issue branch, and re-syncing the base.

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
lists every `Closes #<N>`) finally merges. An omnibus is tracked two ways, kept in
step by `/ship`: its sub-issues are linked as native GitHub **sub-issues** (the
hierarchy / progress panel, which tracks issue *closure* — so under `onto` it reads
0-closed until promotion), and its body carries a machine-anchorable **checklist
block** — a comment-delimited region (`<!-- omnibus-checklist:start -->` … `:end`)
with one `- [ ] #<N>` line per sub-issue, alongside the prose `### Sub-issues`
narrative, tracking the finer-grained *branch-landing* status. As it opens each
sub-PR, Claude ticks that issue's box and ensures its sub-issue link (seeding both
on the bundle's first ship if missing, after showing you the edit), so tracking
stays current instead of drifting, and the omnibus → main PR reads its `Closes` set
straight from the block. The `main`-only guard rail is
unchanged, so the omnibus branch itself isn't hook-protected — keeping work on
sub-PRs is discipline, not enforcement. Composes with the two tokens above.

When the bundle is ready, `/ship #169` — the omnibus issue **on its own, no
`onto`** — opens that omnibus → main PR. Claude recognizes the omnibus issue (a
`vX.Y.Z — …` title with a branch ahead of `main`), skips the implement machinery —
the work already landed on the branch — and drafts the PR with a `Closes #<N>`
line for every sub-issue, cross-checked against the PRs actually merged onto the
branch. You approve and merge it; that one merge closes the whole bundle. Cutting
the release is the later, separate `/release` step.

### Assembling a bundle unattended: `/ultraship`

`/ship #<N> onto #<omnibus>` lands **one** sub-issue at a time, with you in the
loop at every plan and PR PAUSE. `/ultraship` takes a **single omnibus issue** and
assembles the *whole* bundle while you sleep. It has two verbs and exactly one
attended moment:

- **`/ultraship plan #<omnibus>`** — a **director** interviews you to sharpen each
  sub-issue's `objective` and the omnibus `goal`, writes those back into the
  issue's `### Sub-issues` manifest, validates the manifest is well-formed
  (`scripts/ultraship.py`), prints the wave DAG, and **stops.** This is the only
  step that asks you anything.
- **`/ultraship run #<omnibus>`** — fully unattended. A **stage-manager** fans the
  sub-issues out (in dependency waves derived from `depends-on` + `touches`-
  overlap) to **player** subagents, each in its own sibling worktree running the
  `pytest` gate (the `simplify-refactor` half runs at the stage-manager over each
  sub-PR's diff, since a subagent can't spawn another agent); merges their
  "Part of #N" sub-PRs through a **serialized merge train** (conflict or red suite
  → *park*, never resolved unattended); runs a bounded **critic** loop; has the
  director **grade** the result against the goal; and leaves a finished
  `omnibus/vX.Y.Z` branch plus a **risk-ranked morning report** as the
  promotion-PR body.

The manifest, validation, and wave scheduling are deterministic and live in
`scripts/ultraship.py` (pure, unit-tested) — only the orchestration is agentic.
The state is reconstructed from GitHub on every restart (merged sub-PR = done,
open = parked, neither = untouched), so a run that dies at 3 a.m. resumes by
re-reading GitHub rather than replaying a journal.

One safety rule bends **deliberately and only here** (see
[`docs/decisions/GH-0244-…`](./docs/decisions/GH-0244-ultraship-authority-boundary-0001.md),
refined by [`GH-0335-…`](./docs/decisions/GH-0335-ultraship-subagent-players-gate-at-stage-manager-0001.md)):
the stage-manager merges sub-issue → omnibus autonomously (recoverable branch,
serialized, gated, never force-pushed). Players are **subagents** running under
the session's own permission posture, so — unlike the spawned `claude -p` process
GH-0244 first specified — they need no pre-granted permissions and can't hang on a
1 a.m. prompt. The boundary that does **not** move: **a human still merges omnibus
→ `main`** via the `/ship #<omnibus>` promotion mode above, and the
`guard-main-write.sh` hook is untouched.

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
