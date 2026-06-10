# ultraship — unattended omnibus assembly

`/ship #<N> onto #<omnibus>` lands **one** sub-issue on an omnibus branch,
attended. `/ultraship` takes a **single omnibus issue** and assembles the *whole*
bundle: a **director** interviews you up front to sharpen the goal, then the run
goes unattended — a **stage-manager** fans the sub-issues out to **players**,
integrates them through a serialized merge train of sub-PRs, runs a bounded
**critic** review loop, and leaves a finished `omnibus/vX.Y.Z` branch plus a
risk-ranked morning report. **A human still merges omnibus → `main`** via the
existing `/ship` promotion mode — that boundary does not move and the
`guard-main-write.sh` hook is untouched.

This is additive over `/ship onto`: sub-PRs stay **"Part of #N"** with no
`Closes`, merged via `gh pr merge`; the `omnibus-checklist` block, native
sub-issue links, promotion mode, and the guard hook are all untouched; the
per-player gates (`uv run pytest` → `simplify-refactor` → `uv run pytest`) still
run inside each worker.

## The two honest bets

`/ultraship` does **not** "move judgment to the manifest." It makes two specific
bets — name them in the report so a reader can check them:

1. **The up-front director interview front-loads the *requirements* judgment** the
   `/ship` plan PAUSE normally supplies. It is **not** a substitute for
   *implementation* judgment: a surprise that only surfaces against the actual
   code ("the obvious approach violates an invariant") still **parks with a
   written question** at night. That escape valve is the design, not a hole — it
   is how unattended players stay compatible with the repo's *"stop and ask,
   don't wing it"* agreement: front-load the asking that *can* be front-loaded.
2. **An automated critic + a director grade replace the per-diff human review**
   the `/ship` PR PAUSE supplies — backed by per-sub-issue "Part of #N" PRs so
   you can still drill into anything the report flags.

If either bet is wrong, the failure mode is a human rubber-stamping a stacked
diff at 8 a.m. The risk-ranked report exists specifically to fight that.

## The cast

| Role | Owns the question | Where | Suggested model |
|---|---|---|---|
| **director** | *Is it the right work?* — sharpen up front; grade vs `goal` at the end | `plan` interview + run FINISH | a strong reasoning/grading model |
| **stage-manager** | *Sequence + reconcile* — orchestrate, merge, apply small fixes | the orchestrator (you) | a strong coding/agent model |
| **players** | *Implement the sub-issue* | one per sub-issue, own sibling worktree | a strong coding model |
| **critic** | *Is it correct?* — bugs, especially integration | run FINISH, looped | a strong reasoning model |

The director **bookends** the production: shapes it at the start, judges it at
the end — grading against a goal it sharpened *with you*, never one it invented.

**Models are suggested, not pinned.** Pick the best currently-available model for
each role's *job* — a strong reasoning model for the director/critic grading
roles, a strong coding model for the stage-manager/players. As of this writing
**Fable 5** is the strongest model for the judgment roles and **Opus** for the
coding roles, but do not hard-code a model id as a contract: when a better model
ships, prefer it. Pass the choice through `claude -p --model <id>` per role; if
unsure, inherit the session default rather than pin a stale id.

## Two verbs

- **`/ultraship plan #<omnibus>`** — the one attended moment. Director interviews
  you → writes sharpened objectives/goal back into the issue tracking →
  structural validation → prints the wave DAG → **stops.**
- **`/ultraship run #<omnibus>`** — fully unattended. Re-validate → stage-manager
  orchestrates players → merge train → critic loop → director grade → report.

All human judgment collapses into `plan`. `run` is the night.

## The manifest

The manifest **slots into the existing omnibus-tracking regime** (`GH-0297`,
`#265`) — it does not replace it. That regime keeps an omnibus tracked three
ways, all preserved:

- the strict, anchored **`<!-- omnibus-checklist:start/end -->` block** — one line
  per sub-issue (`- [ ] #N — label` → `- [x] #N — label (#sub-PR)`), **which
  promotion mode parses as the `Closes` manifest**;
- native GitHub **sub-issue links** (REST `sub_issues`);
- the narrative **`### Sub-issues`** prose section.

ultraship's metadata lives in the **narrative `### Sub-issues` section** (free
prose the regime preserves), as a strictly-parsed block under each sub-issue's
line, plus one required header line and an optional director's-notes block:

```
**goal:** <a checkable statement of what the assembled bundle achieves>

<!-- director-notes:start -->
X is out of scope. Prefer approach Y for #232.
<!-- director-notes:end -->

### Sub-issues
- #236 — *(bug)* PER_WORKER_GB undercounts parse-worker RSS.
  - **objective:** per-worker estimate tracks measured peak RSS; no swap freeze
  - **touches:** `bartleby/scribe/sizing.py`, `bartleby/scribe/workers.py`
  - **depends-on:** —
```

Three fields do the work — **objective** (one line; the critic/director detect
scope drift against it), **touches** (declared blast radius; the stage-manager
diffs these for overlap), **depends-on** (hard ordering edges, `—` for none).
**Parallelism is *derived*** from `depends-on` + `touches`-overlap — there is no
`parallel` field to contradict the computed answer. The `goal:` header line is
the director's only rubric.

The strict checklist block and the sub-issue links are **left exactly as the
regime defines them.** Because the stage-manager is the *single serial writer* of
those edits, ultraship satisfies `#265`'s reconciler discipline by construction —
the parallel-checkbox-drop `#265` fixed cannot happen here.

### The deterministic tool

Parsing, validation, and wave derivation are **not** left to a model — they live
in `scripts/ultraship.py` (pure, agent-free, unit-tested). The omnibus body is
the input; pipe it in:

```
gh issue view <omnibus> --json body --jq .body | uv run python scripts/ultraship.py validate
gh issue view <omnibus> --json body --jq .body | uv run python scripts/ultraship.py plan
gh issue view <omnibus> --json body --jq .body | uv run python scripts/ultraship.py plan --json
```

`validate` exits non-zero with one readable list of problems if any sub-issue is
missing a field, a `depends-on` doesn't resolve, there's a dependency cycle, a
sub-issue is declared twice, or `goal` is absent. `plan` prints the wave
schedule (`--json` for the machine-readable DAG the run consumes).

---

## `/ultraship plan #<omnibus>`

The semantic counterpart to the structural validator: the interview makes the
manifest *good*; validation checks it's *well-formed*. This is the **only**
attended step.

1. **Preconditions & resolve the branch.** Clean `main`, good `gh auth`. Read the
   omnibus (`gh issue view <omnibus>`); derive its branch from the title's leading
   version (`vX.Y.Z — …` → `omnibus/vX.Y.Z`). If the title has no parseable
   leading `vX.Y.Z`, **refuse** — don't invent a name.
2. **Director interview (the keystone).** Before any DAG, the director reads the
   omnibus and its sub-issues and **talks to you**:
   - asks only what **materially changes the plan or the grade** (not an
     interrogation; you can say "enough, go" anytime);
   - writes the sharpened per-sub-issue **`objective`** and the omnibus **`goal`**
     into the narrative `### Sub-issues` section and omnibus header;
   - records run-wide context that isn't any single issue's objective ("X is out
     of scope," "prefer approach Y for #232") into the **director's-notes block**
     in the omnibus body, so the stage-manager and critic see exactly what the
     director will later grade against. One source of truth, no side-channel.
3. **Seed the tracking if absent.** If the omnibus has no `omnibus-checklist`
   block or no native sub-issue links, propose seeding both (as `/ship onto`'s
   first-ship does) — never silently. Show the proposed body diff and the link
   set before applying.
4. **Structural validation.** Run `scripts/ultraship.py validate`. If it fails,
   show the problems and stop — the manifest is not runnable.
5. **Print the wave DAG** (`scripts/ultraship.py plan`) and **stop.** Do not start
   `run`. `plan` and `run` are separate invocations on purpose.

---

## `/ultraship run #<omnibus>`

Fully unattended. Re-resolve the branch and **re-validate** (`scripts/ultraship.py
validate`) first — a malformed manifest discovered at 2 a.m. after three issues
landed is the worst outcome; abort loud, never guess a plan.

```
0. stage-manager  COLLISION SCAN — one pass at run start for unrelated in-flight
                  work (open PRs against the omnibus, other live worktrees) the
                  DAG can't see. Surface overlaps in the report; the DAG covers
                  intra-bundle ordering only.
1. stage-manager  PLAN — waves from `scripts/ultraship.py plan --json` (topo by
                  depends-on + touches-overlap). Post the wave plan as an omnibus
                  comment.
2. stage-manager  Dispatch a wave -> each sub-issue to a player in its own SIBLING
                  worktree (`../bartleby-issue-<N>-<slug>`, never nested) off the
                  CURRENT omnibus HEAD. Instructions = that issue's objective +
                  the director's-notes.
3. players        Implement; run per-player gates IN ORDER — `uv run pytest` ->
                  `simplify-refactor` -> apply -> `uv run pytest` -> commit
                  (`/ship`'s pytest-skip rules apply). Write their own
                  `docs/decisions/GH-NNNN-*.md` file but NOT the
                  `docs/decisions/README.md` line (see Docs). Push the branch,
                  open a "Part of #N" sub-PR against the omnibus branch (no
                  `Closes`).
4. stage-manager  Merge the sub-PRs serially (a train, never N-way): run the
                  integration suite (full `uv run pytest` on the omnibus branch
                  post-merge; `/ship`'s skip rules apply) — green + clean merge ->
                  `gh pr merge`, tick the checklist block, ensure the native
                  sub-issue link; genuine conflict or red suite -> PARK (leave the
                  sub-PR OPEN, record it). No unattended conflict-resolution.
5. stage-manager  -> critic: review the assembled omnibus.
6. critic         Risk-ranked, plain-language findings (see Report).
7. stage-manager  Triage & assign:
                    small / integration-seam fix -> stage-manager commits it
                      directly to the omnibus (no PR — an "integrator commit"),
                      re-run the suite;
                    substantial / in-sub-issue rework -> a FRESH branch off
                      current omnibus HEAD carrying a rework objective + new
                      sub-PR (the issue's diff now spans two PRs; the report links
                      them).
8.                Loop 5-7 until the critic is clean OR max passes; unresolved ->
                  park.
9. stage-manager  Consolidate all `GH-NNNN` entries into
                  `docs/decisions/README.md` in one pass.
10. director      Grade the assembled bundle vs `goal` + objectives +
                  director's-notes + CLAUDE.md. ADVISORY: if off-target, say so
                  loudly in the report — it does NOT open a second fix loop. You
                  decide at 8 a.m.
11. stage-manager REPORT. Human merges omnibus -> main via `/ship #<omnibus>`
                  promotion mode.
```

**Who fixes what (step 7):** integration-seam and small fixes belong to the
**stage-manager** — it's the integrator, it understands the assembly, and
re-dispatching a player for a one-liner is expensive *and* wrong-footed (omnibus
HEAD has moved under that player). Only "your whole approach is off" goes back to
a player as a fresh-branch rework.

## Players, mechanically

A player is a **headless `claude -p` process** launched in its sibling worktree,
running with permissions pre-granted (an allowlist of git / `gh` / `uv` / the
gate agents, or `--dangerously-skip-permissions`) — otherwise it hangs on the
first permission prompt at 1 a.m. with nobody to approve. This is a deliberate
safety decision (an unsupervised agent with broad command access in a worktree),
recorded in `docs/decisions/`. A blocked or genuinely-unsure player **parks with
a written question** for the report — it never guesses (the up-front interview is
what keeps this rare). On restart the stage-manager adopts or cleans **only its
own** orphaned player worktrees; it never touches a worktree it didn't create.

## Integration & state = sub-PRs + git

Integration is **PR-based** — each landed sub-issue is a "Part of #N" sub-PR the
stage-manager merges via `gh pr merge` (exactly `/ship onto`'s shape, run
unattended and serialized). The sub-PR is *both* the integration mechanism *and*
the drill-down surface, so it must be opened **before** the merge, never after.

The stage-manager keeps **no separate progress store**; it reconstructs the run
from GitHub on every (re)start:

- **merged sub-PR → done** (cross-checked against `git log` of the omnibus branch);
- **open sub-PR → parked / in-flight** (the branch is pushed, the diff reviewable
  — no work is lost to a crash);
- **neither → untouched.**

The checklist block is a **human-readable mirror**, repaired from this on restart
— if a sub-PR merged but the tick failed to write, the merge wins. A parked
issue's box stays `[ ]`, and a stage-manager-maintained **"run status" comment**
on the omnibus names the parked set + reasons (it seeds the morning report). A
run that dies at 3 a.m. resumes by re-reading GitHub, not by replaying a journal.

## The 8 a.m. report

The promotion-PR body is a **risk-ranked report written by the report model**,
prompted explicitly: *"John reads this at 8 a.m. and needs help deciding what to
scrutinize. Describe every red/yellow item in plain language — what changed, why
it's uncertain, what you'd check — not in code-review jargon."* Tiers:

- **🔴 Needs your eyes** — a sub-issue was parked, an *unresolved* critic finding
  remains, or the director flagged scope/goal drift. One plain-language paragraph
  each.
- **🟡 FYI** — landed clean but touched a sensitive seam (shared file, config,
  schema), **or is an *integrator commit*** (a stage-manager step-7 fix with no
  sub-PR of its own — the one category without a drill-down PR, so it's surfaced
  here with its diff).
- **🟢 Clean** — implemented, gates green, merged with no drama. One line each.

The report's job is to *direct attention by risk* — "don't rubber-stamp #232, its
sub-PR is parked on a conflict" — so you drill into the flagged sub-PR instead of
skimming a stacked blob.

## Bounds

- **Max critic passes per run** → then park unresolved findings.
- A **per-player timeout** *and* a whole-run token/time budget — the per-player
  bound stops one runaway wave from eating the entire budget; the stage-manager
  enforces the run budget between waves.
- **Never force-push the omnibus branch** — an unsupervised stage-manager must not
  be able to destroy a night's landed work; fast-forward / merge commits only.

## Authority boundary

- **The stage-manager merges sub-issue → omnibus autonomously** — the omnibus
  branch is recoverable; this is the only place the human-merge rule bends
  (recorded in `docs/decisions/`).
- **A human merges omnibus → `main`**, unchanged, via `/ship #<omnibus>`
  promotion mode. The guard hook protects `main` only; no hook change.

## Docs sweep

Players write their own uniquely-named `docs/decisions/GH-<NNNN>-*.md` (no
conflict) but **do not** touch `docs/decisions/README.md` — N players appending to
the same newest-first line is a guaranteed every-wave conflict the `touches`
field can't predict. The **stage-manager appends all README lines in one pass** at
step 9.

## Out of scope

- Relaxing human-merge-to-`main`.
- Auto-cutting releases (stays the separate `/release` act).
- The director inventing vision — it grades only against `goal` + `objective`s +
  director's-notes + CLAUDE.md agreements.
- Standalone cross-issue dedup — per-player `simplify` covers per-issue hygiene;
  cross-issue duplication is rare and low-stakes.
- Mid-run re-sequencing — conflicts park instead.
- A `parallel` manifest field — parallelism is derived.
- Unattended conflict **auto-resolution** — conflicts park; never resolved
  unsupervised.
