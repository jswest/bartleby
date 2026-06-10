# `/ultraship` bends two safety rules deliberately: the stage-manager merges sub-issue → omnibus autonomously, and players run unattended with pre-granted permissions (issue #244)

> Source: [#244](https://github.com/jswest/bartleby/issues/244)

`/ultraship` assembles a whole omnibus bundle unattended — a stage-manager fans
the sub-issues out to headless players, integrates them through a serialized
merge train, runs a bounded critic loop, and leaves a finished `omnibus/vX.Y.Z`
branch plus a morning report. Running a full bundle while nobody is watching
forces two departures from how the repo otherwise works. Both are deliberate,
bounded, and recorded here so they aren't mistaken for drift.

**1. The stage-manager merges sub-issue → omnibus autonomously.** Everywhere else
the rule is *a human merges* — `/ship`'s PR PAUSE, and the omnibus → `main`
promotion. `/ultraship`'s merge train (`gh pr merge` of each "Part of #N" sub-PR
into the omnibus branch) is the **one** place that rule bends. It is safe to bend
*there* and only there because the omnibus branch is **recoverable**: every
landed sub-issue is a pushed branch and an open-then-merged PR, the train is
**serialized** (never an N-way merge), the integration suite gates each merge
(red suite or genuine conflict → **park**, leave the sub-PR open, resolve nothing
unattended), and the branch is **never force-pushed** — fast-forward / merge
commits only, so a night's landed work cannot be destroyed. The boundary that
does *not* move: **a human merges omnibus → `main`**, via the existing `/ship
#<omnibus>` promotion mode. The `guard-main-write.sh` hook protects `main` only
and is **untouched** — `/ultraship` adds no hook change and reaches around no
guard.

**2. Players run unattended with pre-granted permissions.** A player is a headless
`claude -p` process in its own sibling worktree. An unattended agent that hits a
permission prompt at 1 a.m. with nobody to approve simply **hangs** — so players
launch with permissions pre-granted (an allowlist of git / `gh` / `uv` / the gate
agents, or `--dangerously-skip-permissions`). Granting an unsupervised agent
broad command access in a worktree is a real exposure; it is accepted because the
blast radius is a throwaway sibling worktree and a recoverable omnibus branch
(never `main`), and the worktree-discipline still holds — the stage-manager
adopts or cleans **only its own** orphaned player worktrees, never one it didn't
create. A blocked or genuinely-unsure player **parks with a written question**
for the morning report rather than guessing; the up-front director interview
exists to make that rare by front-loading the *requirements* judgment that can be
front-loaded.

**Why these are not a softening of the repo's agreements.** The "stop and ask,
don't wing it" rule is honored by *moving the asking earlier*, not deleting it:
the director interview is the one attended moment, and the park-with-a-question
escape valve covers what only surfaces against the code. The single-serial-writer
discipline that `#265`/`GH-0297` require of omnibus tracking is satisfied **by
construction** — the stage-manager is the sole writer of the checklist block and
sub-issue links, so the parallel-clobber `#265` fixed cannot occur. State lives in
GitHub (merged PR = done, open = parked, neither = untouched), so a crash resumes
by re-reading GitHub, not by trusting a journal.

Skill + tooling change — `.claude/skills/ultraship/SKILL.md` plus the pure
manifest/validate/DAG core in `scripts/ultraship.py` (unit-tested, agent-free).
No `SCHEMA_VERSION` bump, no product code path, no hook change.
