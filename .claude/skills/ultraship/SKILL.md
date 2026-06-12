---
name: ultraship
description: >-
  Assemble a whole omnibus bundle unattended from a single `Omnibus:`-titled
  issue: a director interview up front, then a stage-manager fans sub-issues to
  player subagents, integrates them through a serialized merge train of "Part of
  #N" sub-PRs, runs a bounded critic loop, and auto-opens the omnibus→base PR
  carrying a risk-ranked morning report. A human still merges. Portable and
  config-driven via `.claude/ultraship.toml`. Three verbs: `/ultraship config`
  (guided setup), `/ultraship plan #<omnibus>` (attended interview), `/ultraship
  run #<omnibus>` (unattended night).
---

# ultraship — portable, unattended omnibus assembly

`/ultraship` takes a **single omnibus issue** and assembles the *whole* bundle: a
**director** interviews you up front to sharpen the goal, then the run goes
unattended — a **stage-manager** fans the sub-issues out to **players**,
integrates them through a serialized merge train of "Part of #N" sub-PRs, runs a
bounded **critic** review loop, **auto-opens the omnibus→base PR** carrying a
risk-ranked morning report, and stops. **A human still merges omnibus → the base
branch** — that boundary does not move.

This skill is **portable**: one source lives in the drawer at
`~/Code/workflows/skills/ultraship/` and is *pressed* by `signet` into each
repo's `.claude/skills/ultraship/`. Everything repo-specific — the test command,
the quality-gate agent, the branch scheme, the decision-log paths, the live-data
redline — lives in a per-repo `.claude/ultraship.toml` that the skill **detects
once and persists**. The skill prose below never hard-codes a repo's specifics;
it reads them from config.

## The two honest bets

`/ultraship` does **not** "move judgment to the manifest." It makes two specific
bets — name them in the report so a reader can check them:

1. **The up-front director interview front-loads the *requirements* judgment** a
   normal attended ship PAUSE supplies. It is **not** a substitute for
   *implementation* judgment: a surprise that only surfaces against the actual
   code ("the obvious approach violates an invariant") still **parks with a
   written question** at night. That escape valve is the design, not a hole — it
   is how unattended players stay compatible with a repo's *"when unsure, stop and
   ask"* agreement: front-load the asking that *can* be front-loaded.
2. **An automated critic + a director grade replace the per-diff human review** —
   backed by per-sub-issue "Part of #N" PRs so you can still drill into anything
   the report flags.

If either bet is wrong, the failure mode is a human rubber-stamping a stacked diff
at 8 a.m. The risk-ranked report exists specifically to fight that.

## The cast

| Role | Owns the question | Where | Suggested model |
|---|---|---|---|
| **director** | *Is it the right work?* — sharpen up front; grade vs `goal` at the end | `plan` interview + run FINISH | a strong reasoning/grading model |
| **stage-manager** | *Sequence + reconcile* — orchestrate, merge, apply small fixes | the orchestrator (you) | a strong coding/agent model |
| **players** | *Implement the sub-issue* | one subagent per sub-issue, own sibling worktree | a strong coding model |
| **critic** | *Is it correct?* — bugs, especially integration | run FINISH, looped | a strong reasoning model |

The director **bookends** the production: shapes it at the start, judges it at the
end — grading against a goal it sharpened *with you*, never one it invented.

**Models are suggested, not pinned.** Pick the best currently-available model for
each role's *job* — a strong reasoning model for the director/critic grading
roles, a strong coding model for the stage-manager/players. As of this writing
**Fable 5** is strongest for the judgment roles and **Opus** for the coding roles,
but do not hard-code a model id as a contract: when a better model ships, prefer
it. Players are dispatched as subagents (below), so pass their model through the
Agent tool's `model` parameter; pass any other role's model through its own
launch. If unsure, inherit the session default rather than pin a stale id.

## Three verbs

- **`/ultraship config`** — guided `.claude/ultraship.toml` creation/edit. Runs
  automatically on a repo's first `plan`/`run`. Attended.
- **`/ultraship plan #<omnibus>`** — the director interview + structural
  validation + wave DAG, then stop. The one attended planning moment.
- **`/ultraship run #<omnibus>`** — the unattended night.

All human judgment collapses into `config` (once per repo) and `plan` (once per
omnibus). `run` is the night.

---

## Title & membership conventions (`Omnibus:` / `Issue:`)

- **An omnibus issue** is titled `Omnibus: <rest>`. `/ultraship` **refuses** to act
  on an issue whose title does not start with `Omnibus:` — it will not invent a
  bundle from a normal issue. This is the **only** hard title requirement.
- **A sub-issue** is titled `Issue: <rest>`. The skill *warns* (does not hard-fail)
  if a manifest member isn't `Issue:`-titled — the manifest is authoritative for
  membership; the title prefix is a sanity check.
- **Membership + ordering come from the curated `### Sub-issues` manifest** in the
  omnibus body, not from titles. Native GitHub sub-issue links are maintained as a
  parallel hierarchy layer, but the manifest block is what the deterministic tool
  parses and what waves derive from.
- **Slug** = the omnibus/sub-issue title with the `Omnibus:`/`Issue:` prefix
  stripped, kebab-cased, truncated. Used for branch and worktree names.

### Branch naming (version-optional)

`/ultraship` **never refuses for lack of a version**. The branch is resolved by
this precedence:

1. **Versioned (auto when supported).** When `version_scheme` is set in config
   *and* a `vX.Y.Z` token is parseable from the omnibus title (after the
   `Omnibus:` prefix — e.g. `Omnibus: v0.9.7 — …`), the branch is
   `<branch_prefix>/vX.Y.Z` (e.g. `omnibus/v0.9.7`).
2. **Fallback (default).** Otherwise the branch is `<branch_prefix>-<slug>`
   (e.g. `omnibus-amount-range-fixes`) — slug from the omnibus title minus its
   `Omnibus:` prefix. If that name already exists for a *different* omnibus,
   disambiguate by appending the issue number (`<branch_prefix>-<slug>-<N>`).

A version is a bonus a project *earns* (flip `version_scheme` on later, no other
change), never a gate. Only a missing `Omnibus:` prefix is a refusal.

---

## Per-project config — `.claude/ultraship.toml`

Each repo's specifics live in `.claude/ultraship.toml`. First run **detects and
proposes**; on approval it **persists**; later runs **read it directly**. The skill
never reaches for a repo's specifics from anywhere else.

```toml
# .claude/ultraship.toml — written by `/ultraship config`, editable by hand.
base_branch      = "main"                        # promotion target (default branch)
branch_prefix    = "omnibus"                      # see "Branch naming"
version_scheme   = ""                             # "" = always {prefix}-{slug};
                                                  #   "leading-v" = prefer {prefix}/vX.Y.Z
                                                  #   when a version is parseable from the title.
worktree_pattern = "../{repo}-issue-{n}-{slug}"  # sibling, never nested

test_cmd         = "uv run pytest"               # detected; "" = no suite (rare; warn)
gate_agent       = "simplify-refactor"           # quality agent run at stage-manager over each
                                                  #   sub-PR diff; "" = skip that gate half
                                                  #   (the run warns once in the report).
ship_skill       = "ship"                         # per-issue skill if present (informational)

# Decision log — ENCOURAGED but OPTIONAL. Leave both "" to skip the decision-file
# + index steps entirely, with no nagging. Set them to opt in.
decisions_dir    = "docs/decisions"              # "" = project keeps no decision log
decisions_index  = "docs/decisions/README.md"    # "" = no newest-first index to maintain

guard_hook       = ".claude/hooks/guard-main-write.sh"  # detected; informational only
docs_only_globs  = ["*.md", "LICENSE*", "docs/**"]      # auto-skip the suite when wholly docs
skip_tests_token = true                                 # honor a `skip-tests` opt-in

# Free text injected VERBATIM into every player's instructions — how a repo states
# its "never touch live X" redline without the global skill knowing the specifics.
#   e.g. "Stay offline. Verify only against tests/fixtures/. Never fetch the live site."
#   e.g. "Never drive serve/ingest/upgrade against the live home; use a throwaway sandbox."
player_guardrails = ""
```

### Detection heuristics (first run / `--reconfigure`)

- `test_cmd`: `pyproject.toml` + `uv.lock` → `uv run pytest`; `package.json` with a
  `test` script → `npm test` (or `pnpm`/`yarn` per lockfile); `Cargo.toml` →
  `cargo test`; `go.mod` → `go test ./...`; a `Makefile` `test:` target → `make
  test`. Ambiguous/none → ask in the interview.
- `gate_agent`: probe the agent registry for `simplify-refactor` (or the repo's
  documented quality agent); if none, leave empty and skip that gate half.
- `decisions_dir`/`index`: present iff the path exists; absent → left `""`.
- `version_scheme`: `"leading-v"` iff the repo already versions — git tags shaped
  `vX.Y.Z`, a `__version__`/`version =` in packaging, or an omnibus title carrying
  a `vX.Y.Z`; else `""`. Never required.
- `guard_hook`: scan `.claude/settings*.json` PreToolUse hooks + `.claude/hooks/`.
- `base_branch`: the repo's default branch
  (`gh repo view --json defaultBranchRef --jq .defaultBranchRef.name`).
- Everything detected is **shown for approval**, never silently written.

### `/ultraship config` — the wizard

The first `plan`/`run` in a repo with **no** `.claude/ultraship.toml` **routes into
the wizard first** (it does not silently autogenerate). `/ultraship config` invokes
it on demand; `--reconfigure` re-runs it over an existing file. It is a short
guided interview, not a form dump:

1. Run detection (above) to pre-fill every field.
2. Walk the fields in plain language, each with its detected value as the default
   the user accepts with a keystroke or overrides:
   - "Tests look like `uv run pytest` — right?" (`test_cmd`)
   - "Quality gate agent `simplify-refactor` is available — run it over each
     sub-PR diff?" (`gate_agent`)
   - "This repo tags releases `vX.Y.Z` — name omnibus branches `omnibus/vX.Y.Z`?
     (else `omnibus-<slug>`)" (`version_scheme`)
   - "Keep a decision log? I found `docs/decisions/` — **encouraged**; want
     `/ultraship` to write ADR files + maintain the index? (skip with no penalty)"
     (`decisions_dir`/`index`)
   - "Any live-data redline a night-time player must never cross?"
     (`player_guardrails` — free text)
3. Show the assembled `.toml` and **write it on approval.** Done once per repo;
   every later `plan`/`run` reads it directly.

**Config commit policy: commit it.** The `.toml` holds facts about the repo and
how the team ships it — base branch, test command, gate agent, decision paths, and
the live-data redline — identical for everyone on the repo, so it belongs in git.
The redline (`player_guardrails`) especially **must** be shared: a machine-local
config means a teammate's checkout, a fresh clone, or a CI run gets an *empty*
guardrail and an unattended player could cross the exact line the config exists to
prevent. Committing also makes a branch-scheme or test-command change reviewable in
a PR, and keeps later runs deterministic. The wizard does not ask (the interview
stays short); the rare repo that wants it machine-local hand-adds one `.gitignore`
line.

---

## The manifest

The manifest **slots into the omnibus-tracking regime** an attended `ship onto`
flow maintains — it does not replace it. That regime keeps an omnibus tracked three
ways, all preserved:

- the strict, anchored **`<!-- omnibus-checklist:start/end -->` block** — one line
  per sub-issue (`- [ ] #N — label` → `- [x] #N — label (#sub-PR)`), which
  promotion parses as the `Closes` manifest;
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
- #236 — *(bug)* <one-line label>
  - **objective:** <one line; the critic/director grade scope against this>
  - **touches:** `path/a.py`, `path/b.py`   # declared blast radius → overlap serializes
  - **depends-on:** —                        # hard ordering edges; — = none
```

Three fields do the work — **objective** (one line; the critic/director detect
scope drift against it), **touches** (declared blast radius; the stage-manager
diffs these for overlap), **depends-on** (hard ordering edges, `—` for none).
**Parallelism is *derived*** from `depends-on` + `touches`-overlap — there is no
`parallel` field to contradict the computed answer. The `goal:` header line is the
director's only rubric.

The strict checklist block and sub-issue links are **left exactly as the regime
defines them.** Because the stage-manager is the *single serial writer* of those
edits, the parallel-checkbox-drop hazard cannot happen here — no concurrent writer
races on the omnibus body.

### The deterministic tool

Parsing, validation, and wave derivation are **not** left to a model — they live in
the pressed `.claude/skills/ultraship/ultraship.py` (pure stdlib, agent-free, no
git/gh side effects). It is run with **system `python3` by absolute path** — no
`uv`, no project venv, no deps — so the one shared tool runs identically in every
repo. The omnibus body is the input; pipe it in:

```
gh issue view <omnibus> --json body --jq .body | python3 .claude/skills/ultraship/ultraship.py validate
gh issue view <omnibus> --json body --jq .body | python3 .claude/skills/ultraship/ultraship.py plan
gh issue view <omnibus> --json body --jq .body | python3 .claude/skills/ultraship/ultraship.py plan --json
```

`validate` exits non-zero with one readable list of problems if any sub-issue is
missing a field, a `depends-on` doesn't resolve, there's a dependency cycle, a
sub-issue is declared twice, an unrecognized line appears in the section, or `goal`
is absent. `plan` prints the wave schedule (`--json` for the machine-readable DAG
the run consumes). If the repo genuinely lacks `python3`, that's a preflight
failure.

---

## `/ultraship plan #<omnibus>`

The semantic counterpart to the structural validator: the interview makes the
manifest *good*; validation checks it's *well-formed*. This is the **only** attended
planning step.

1. **Preflight + resolve config.** Clean `base_branch`, good `gh auth`, `python3`
   present. Load `.claude/ultraship.toml`; **if absent, route into the config
   wizard first** and write it on approval before continuing.
2. **Resolve the branch.** Read the omnibus (`gh issue view <omnibus>`). Title must
   start `Omnibus:` else **refuse**. Derive the branch by the version-optional
   precedence: `<branch_prefix>/vX.Y.Z` when `version_scheme` is set and a version
   is parseable, else `<branch_prefix>-<slug>`. **Lack of a version is never a
   refusal** — only a missing `Omnibus:` prefix is.
3. **Director interview — reads bodies *and* comment threads.** Before any DAG, the
   director reads the omnibus and **every** sub-issue with
   `gh issue view <N> --comments` (a "do it this way" correction usually lands in a
   comment *after* filing), then **talks to you**:
   - asks only what **materially changes the plan or the grade** (not an
     interrogation; you can say "enough, go" anytime);
   - **folds comment corrections into the manifest** — players receive only
     `objective` + director's-notes and never see raw threads, so anything material
     from a comment must be lifted into that issue's `objective`;
   - writes the sharpened per-sub-issue **`objective`** and the omnibus **`goal`**
     into the `### Sub-issues` section / header;
   - records run-wide context ("X out of scope", "prefer approach Y for #232") into
     the **director-notes block**. One source of truth, no side-channel.
4. **Seed tracking if absent.** If the omnibus has no `omnibus-checklist` block or
   no native sub-issue links, propose seeding both (show the body diff + link set);
   never silent.
5. **Structural validation.** `python3 .claude/skills/ultraship/ultraship.py
   validate`; on failure show the problems and **stop** — the manifest isn't
   runnable.
6. **Print the wave DAG** (`… ultraship.py plan`) and **stop.** `plan` and `run` are
   separate invocations on purpose.

---

## `/ultraship run #<omnibus>`

Fully unattended. Re-resolve the branch and **re-validate** first — a malformed
manifest discovered at 2 a.m. after three issues landed is the worst outcome; abort
loud, never guess a plan.

```
0. stage-manager  PREFLIGHT — config loaded, gh auth, python3, clean base.
                  COLLISION SCAN — open PRs against the omnibus + other live
                  worktrees the DAG can't see; surface overlaps in the report.
1. stage-manager  PLAN — waves from `ultraship.py plan --json` (topo by depends-on
                  + touches-overlap). Post the wave plan as an omnibus comment.
2. stage-manager  Dispatch a wave → each sub-issue to a player subagent in its own
                  SIBLING worktree (`worktree_pattern`, never nested) off the
                  CURRENT omnibus HEAD. Instructions = that issue's objective +
                  director's-notes + config.player_guardrails.
3. players        Implement; run `{test_cmd}` → commit (docs-only / skip-tests
                  rules from config apply). If `decisions_dir` set, write their own
                  uniquely-named decision FILE but NOT the index line. Push the
                  branch, open a "Part of #<omnibus>" sub-PR (base = omnibus branch,
                  no `Closes`). (A subagent can't spawn an agent, so the
                  `{gate_agent}` half runs at the stage-manager in step 4.)
4. stage-manager  MERGE TRAIN (serial, never N-way): run `{gate_agent}` over the
                  sub-PR diff and push accepted fixes onto its branch (if
                  `gate_agent` is "", skip this half and note it once for the
                  report — see below); run the
                  integration `{test_cmd}` on the post-merge omnibus branch —
                  green + clean → `gh pr merge`, tick the checklist line, ensure
                  the native sub-issue link; conflict or red → PARK (leave the
                  sub-PR OPEN, record it). No unattended conflict resolution.
5. stage-manager  → critic: review the assembled omnibus.
6. critic         Risk-ranked, plain-language findings (see Report).
7. stage-manager  Triage & assign:
                    small / integration-seam fix → stage-manager commits it
                      directly to the omnibus (no PR — an "integrator commit"),
                      re-run the suite;
                    substantial / in-sub-issue rework → a FRESH branch off current
                      omnibus HEAD carrying a rework objective + new sub-PR.
8.                Loop 5–7 until the critic is clean OR max passes; unresolved →
                  park.
9. stage-manager  If `decisions_index` set, consolidate all new decision entries
                  into the index in one pass.
10. director      Grade the assembled bundle vs goal + objectives + director-notes
                  + the repo's CLAUDE.md. ADVISORY — loud in the report, opens no
                  second loop. You decide at 8 a.m.
11. stage-manager REPORT + AUTO-OPEN PROMOTION PR. Then STOP — a human merges.
```

**Who fixes what (step 7):** integration-seam and small fixes belong to the
**stage-manager** — it's the integrator, it understands the assembly, and
re-dispatching a player for a one-liner is expensive *and* wrong-footed (omnibus
HEAD has moved under that player). Only "your whole approach is off" goes back to a
player as a fresh-branch rework.

## Players, mechanically

A player is an **isolated subagent** the stage-manager dispatches (the Agent tool
with `isolation: worktree`), running in its own sibling worktree off the current
omnibus HEAD. Because a subagent runs inside the stage-manager's own session under
your login, it needs no separate auth and no `--dangerously-skip-permissions`: it
inherits the session's permission posture and can't hang on a fresh 1 a.m.
permission prompt. The trade-off is that a subagent **can't spawn another agent**,
so the `{gate_agent}` half of the gate runs at the stage-manager (run step 4), not
inside the player. A blocked or genuinely-unsure player **parks with a written
question** for the report — it never guesses (the up-front interview is what keeps
this rare). On restart the stage-manager adopts or cleans **only its own** orphaned
player worktrees; it never touches a worktree it didn't create.

**`config.player_guardrails` is injected verbatim into every player prompt.** This
is the generalized home for a repo's live-data redline — "never fetch the live
site", "never touch the live home directory", whatever it is. The global skill
stays ignorant of the specifics; each repo states its redline once, in config. If a
step seems to need crossing that line, that's a **park-with-a-question**, not a
reach across it.

## Integration & state = sub-PRs + git

Integration is **PR-based** — each landed sub-issue is a "Part of #N" sub-PR the
stage-manager merges via `gh pr merge`, serialized. The sub-PR is *both* the
integration mechanism *and* the drill-down surface, so it must be opened **before**
the merge, never after.

The stage-manager keeps **no separate progress store**; it reconstructs the run
from GitHub on every (re)start:

- **merged sub-PR → done** (cross-checked against `git log` of the omnibus branch);
- **open sub-PR → parked / in-flight** (the branch is pushed, the diff reviewable —
  no work is lost to a crash);
- **neither → untouched.**

The checklist block is a **human-readable mirror**, repaired from this on restart —
if a sub-PR merged but the tick failed to write, the merge wins. A parked issue's
box stays `[ ]`, and a stage-manager-maintained **"run status" comment** on the
omnibus names the parked set + reasons (it seeds the morning report). A run that
dies at 3 a.m. resumes by re-reading GitHub, not by replaying a journal. **Never
force-push the omnibus branch** — fast-forward / merge commits only; an unsupervised
run must not be able to destroy a night's landed work.

## The 8 a.m. report → the promotion PR body

At step 11 the stage-manager **builds the `Closes` enumeration** — the `#<N>` set
from the omnibus checklist block, **reconciled against the sub-PRs actually merged
into the branch** (`gh pr list --base <omnibus-branch> --state merged` ⨯
`git log <base>..<omnibus-branch>`); a checklist/merged disagreement is *reported*,
never silently resolved — then **auto-opens the omnibus→base PR**:

```
gh pr create --base {base_branch} --head {omnibus-branch} \
  --title "Omnibus: <rest> → {base_branch}" --body <the morning report>
```

The PR body **is** the risk-ranked report, written by the report model, prompted:
*"John reads this at 8 a.m. and needs help deciding what to scrutinize. Plain
language — what changed, why it's uncertain, what you'd check — not code-review
jargon."* Tiers:

- 🔴 **Needs your eyes** — a parked sub-issue, an *unresolved* critic finding, or
  director-flagged scope/goal drift. One plain-language paragraph each, linking the
  sub-PR.
- 🟡 **FYI** — landed clean but touched a sensitive seam (shared file, config,
  schema), **or is an *integrator commit*** (a step-7 stage-manager fix with no
  sub-PR of its own — the one category lacking a drill-down surface, so its diff
  goes here).
- 🟢 **Clean** — implemented, gates green, merged with no drama. One line each.

If `gate_agent` is unset for this repo, **the report says so once** — a single line
noting that no automated quality gate ran over the sub-PR diffs, so per-sub-PR
hygiene wasn't auto-checked and the human reviewer should weigh that. It's a
one-time notice, not a per-issue nag.

A `Closes #<N>` footer enumerates the bundle. Then **STOP — the human merges**; the
merge to the base branch auto-closes every sub-issue and the omnibus issue.

**Why auto-open is safe and still inside the boundary:** opening a PR neither
commits nor pushes to `base_branch`, so the `guard_hook` (which guards the default
branch) never fires and needs no change. The autonomy added is exactly "draft the PR
+ assemble the report"; the irreversible act — merge to the base branch — stays a
human click. The all-green-only and auto-merge variants were considered and
**declined**: always open so the report is in your face; never auto-merge.

## Bounds & authority

- **Max critic passes per run** → then park unresolved findings.
- A **per-player timeout** *and* a whole-run token/time budget — the per-player
  bound stops one runaway wave from eating the budget; the stage-manager enforces
  the run budget between waves.
- **Never force-push the omnibus branch.**
- **The stage-manager merges sub-issue → omnibus autonomously** — the omnibus branch
  is recoverable; this is the only place the human-merge rule bends.
- **A human merges omnibus → `base_branch`** — unchanged; the skill now *opens* that
  PR but does not merge it.

## Self-check against the drawer (signet)

This skill is pressed from a drawer by `signet`, which leaves a `.stamp`
(`source`, `pressed_at`, per-file `sha256`) beside the pressed files. Honor it:

- **Attended verbs (`config`, `plan`)** read `.claude/skills/ultraship/.stamp`, find
  the drawer path it names, and byte-compare the pressed files. **Drift → stop and
  prompt to re-press** (`signet press ultraship --into <repo>`).
- **If the drawer path in `.stamp` doesn't exist** on this machine → **skip
  silently** (a clone, CI, or a teammate without the drawer is never penalized).
- **The unattended `run`** verb **warns, never aborts** on drift — housekeeping must
  not kill a night's work.
- A **`--no-signet-check`** flag bypasses the check on any verb.

## Out of scope

- Relaxing human-merge-to-base-branch (auto-merge explicitly declined).
- Auto-cutting releases (a separate, later act on the base branch).
- The director inventing vision — it grades only against goal + objectives +
  director-notes + the repo's CLAUDE.md.
- Standalone cross-issue dedup — the per-sub-PR `{gate_agent}` pass covers per-issue
  hygiene.
- Mid-run re-sequencing or unattended conflict **auto-resolution** — conflicts park.
- A `parallel` manifest field — parallelism stays derived.
- Inferring a repo's live-data redline — that's `player_guardrails`, authored by the
  project, not guessed by the skill.
