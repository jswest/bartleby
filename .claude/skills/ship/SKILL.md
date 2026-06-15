---
name: ship
description: >-
  Land one or more issue-branches onto a target branch, tested and reviewed,
  and open the PRs — promoting to `main` when the target is a finished omnibus.
  Invoke as `/ship #<issue>` (optionally `--onto <branch>`). A leaf issue is
  implemented by one player; an issue with native GitHub sub-issues is an
  *omnibus* — fanned out across players, assembled, critiqued, and promoted.
  Portable and config-driven via `.claude/ship.toml`. Use whenever asked to
  ship, implement, or do a numbered issue.
---

# ship — land issue-branches onto a target, tested and reviewed

**One idea underneath everything:** take one or more issue-branches, land them
onto a **target** branch (opening the issue→target PR for each), and — if the
target isn't `main` and it's *finished* — promote it with a target→`main` PR.
A single issue onto `main` is the simple case: a set of one, no promotion. An
**omnibus** is the general case: a parent issue whose native GitHub sub-issues
are the set, assembled on an omnibus branch, then promoted.

## The cast

| Role | Owns | Where | Tier |
|---|---|---|---|
| **director** | plan, dispatch, triage, assemble, promote | the session (you) | strong |
| **players** | implement one issue — code, test, self-tidy, commit, PR | subagents, in parallel, each its own sibling worktree | cheap |
| **critics** | review the work — a **correctness critic** and a **simplicity critic** | subagents the director spools | strong (per-unit simplicity: cheap) |

The director runs the show — plan, dispatch, triage, assemble, promote — and
never grinds: the implementation churn (big reads, test loops, edits) lives in
players' throwaway contexts, so the session stays lean. Players and critics are
subagents the director launches.

## Invocation

Argument is one GitHub issue number (`#118`, `118`). The issue/omnibus fork is a
single API check: **an issue with honest GitHub sub-issues is an omnibus**;
anything else is a leaf. Checklists and title conventions do **not** count —
only native sub-issues (see *Resolve target and argument* for the exact API).

Flags (all optional):

- **`--onto <branch>`** — the target to land on. Defaults to `base_branch`
  (`main`). Landing a leaf onto an existing omnibus branch adds to an in-flight
  omnibus by hand.
- **`--plan`** — force the guidance gate before any work (see *The plan gate*).
  Without it, the director uses judgment about whether to ask — and for an
  omnibus fan-out, leans hard toward asking.
- **`--skip-tests`** — force-skip the suite even for a test-affecting diff.
  Without it, tests run *except* when the diff is wholly within `auto_skip_globs`
  (docs, html/css/ts).
- **`--with-playwright`** — permit a player to drive a real browser via the
  Playwright MCP when it judges it useful. Available only if `.claude/ship.toml`
  declares a `[playwright]` block; otherwise the flag is unknown — say so and
  ignore it.
- **`--strong`** — every role on the strong tier, overriding config.
- **`--thrifty`** — every role on the cheap tier, overriding config.
  (`--strong` and `--thrifty` are mutually exclusive; passing both is an error.)
- **`config` / `--reconfigure`** — run (or re-run) config detection and write
  `.claude/ship.toml` (see *Detection & first run*). `config` implements nothing.
- **`--no-signet-check`** — skip the pressed-skill stamp comparison (see
  *Signet self-check*).

## The two PR events

1. **issue → target** — every time an issue-branch lands. Always.
2. **target → `main`** — *once*, when the target is a non-`main` branch **and
   it's assembled** (the omnibus is complete). This is the **promotion**.

A leaf onto `main` fires only event 1. A leaf `--onto` an omnibus fires only
event 1 — the omnibus isn't done, so its promotion waits. An omnibus fires
event 1 per sub-issue and event 2 once, at the end. The human owns the only
irreversible act — merging a PR to `main`.

## Config — `.claude/ship.toml`

Portable: one source lives at `~/Code/workflows/skills/ship/` and is *pressed* by
`signet` into each repo's `.claude/skills/ship/`. Everything repo-specific lives
in `.claude/ship.toml`, a sibling of the pressed dir that `signet` never touches.
First run **detects and proposes**; on approval it **persists**; later runs
**read it directly**. The prose below never hard-codes a repo's specifics.

```toml
# .claude/ship.toml — written by `/ship config` on first run, editable by hand. Commit it.
base_branch       = "main"                          # ultimate promotion target / default --onto
branch_prefix     = "omnibus"                       # omnibus branch stem
worktree_pattern  = "../{repo}-issue-{n}-{slug}"    # sibling, never nested

test_cmd          = "uv run pytest"                 # "" = no suite (rare; warn)
test_source_globs = ["*.py", "pyproject.toml"]      # a diff touching any of these is "test-affecting"
auto_skip_globs   = ["*.md", "docs/**", "*.html", "*.css", "*.ts"]  # diff wholly within these → tests auto-skip

architecture_doc  = "ARCHITECTURE.md"               # current-state doc swept before promotion ("" = none)
decisions_dir     = "docs/decisions"                # "" = no decision log
decisions_index   = "docs/decisions/README.md"      # "" = no index to maintain

guardrails        = ""                              # free text injected VERBATIM into every player prompt AND
                                                    #   the director's own work — the repo's redline, e.g.
                                                    #   "never fetch the live site", "never touch ~/". "" = none.
guard_hook        = ".claude/hooks/guard-main-write.sh"  # detected; informational only

player_tier       = "cheap"                         # default model tier for players: "cheap" | "strong"
max_critic_passes = 3                               # critic → triage loop cap before parking

[playwright]                                        # optional; present ⇒ --with-playwright is available
frontend_glob = "web/**"                            # the flag fires only if a player's diff touches this
dev_server    = "npm run dev"                       # launched in the worktree
url           = "http://localhost:3000"
```

### Detection & first run

A `/ship #<issue>` in a repo with **no** `.claude/ship.toml` **routes into config
first** — it never silently autogenerates. `/ship config` invokes it on demand;
`--reconfigure` re-runs it over an existing file. Auto-derive every field, ask
only where a value is genuinely ambiguous, show the assembled `.toml`, and
**write it on approval**. Done once per repo.

- `base_branch` — the repo's default branch
  (`gh repo view --json defaultBranchRef --jq .defaultBranchRef.name`).
- `worktree_pattern` — default `../{repo}-issue-{n}-{slug}`, `{repo}` = basename
  of the main checkout (`git rev-parse --show-toplevel`).
- `test_cmd` / `test_source_globs` — from project type: `pyproject.toml`+`uv.lock`
  → `uv run pytest` / `["*.py","pyproject.toml"]`; `package.json` with a `test`
  script → `npm test` (or `pnpm`/`yarn` per lockfile) / `["*.ts","*.js","package.json"]`;
  `Cargo.toml` → `cargo test` / `["*.rs","Cargo.toml"]`; `go.mod` → `go test ./...`;
  a `Makefile` `test:` target → `make test`. Ambiguous / none → **ask**, don't guess.
- `auto_skip_globs` — the docs/asset set whose diffs can't affect the suite
  (default `["*.md","docs/**","*.html","*.css","*.ts"]`).
- `architecture_doc` — the repo's current-state design doc if one exists
  (`ARCHITECTURE.md`, `SPEC.md`, `DESIGN.md`); else `""`.
- `decisions_dir` / `decisions_index` — present iff the path exists; absent → `""`.
- `guard_hook` — scan `.claude/settings*.json` PreToolUse hooks + `.claude/hooks/`.
- `guardrails` — leave `""` unless the repo has a known redline; never invent one.
- `[playwright]` — propose a block only if the repo has a frontend tree (a
  `package.json` dev script under a `web/`-like dir); otherwise omit it (the flag
  stays unavailable). Keep this table **last** — in TOML every bare key after a
  `[table]` header belongs to that table.

**Commit the `.toml`.** It holds facts about the repo and how it ships — base
branch, test command, doc paths, the redline — identical for everyone, so it
belongs in git; the `guardrails` redline especially must be shared so a fresh
clone or CI run never gets an empty guardrail. The rare repo that wants it
machine-local hand-adds one `.gitignore` line.

## Signet self-check

This skill is pressed from a drawer by `signet`, which leaves a `.stamp`
(`source`, `pressed_at`, per-file `sha256`) beside the pressed files. `/ship` is
an **attended** verb, so honor the stamp before doing anything:

- Read `.claude/skills/ship/.stamp`, find the drawer path it names, and
  byte-compare the pressed files. **Drift → stop and prompt to re-press**
  (`signet press ship --into <repo>`) before shipping.
- **If the drawer path in `.stamp` doesn't exist** on this machine → **skip
  silently** (a clone, CI, or a teammate without the drawer is never penalized).
- **`--no-signet-check`** bypasses the check.

## Model tiering

Two tiers, resolved **flag > config > default**:

- **director** — strong.
- **players** — cheap by default (`player_tier`); there are many of them, and
  that's where token spend concentrates.
- **critics** — strong, except the simplicity critic's **per-unit** pass (a
  focused review of one small diff), which is cheap.
- **`--strong`** lifts everything to strong; **`--thrifty`** drops everything to
  cheap.

Pass each subagent's resolved model through the **Agent tool's `model`
parameter**; a player left to inherit the session model runs expensive and
defeats the tiering. As of this writing **Opus** is the strong tier and
**Sonnet** a solid cheap tier — don't hard-code a model id; prefer a better one
when it ships.

---

# The director's flow (you)

## 0. Preconditions
Config loaded (detect+persist on first run; see *Detection & first run*). Signet
self-check honored. `gh` authed. Working tree of the base is clean. Honor
`guardrails` verbatim throughout. `guard_hook`, if set, is informational — the
worktree discipline below is what keeps work off the base.

## 1. Resolve target and argument
Target = `--onto <branch>` if given, else `base_branch`. Fetch the issue **and
its comments** (clarifications, scope changes, and "actually don't do X"
corrections live there — read them every time, not just at the plan gate); check
for **native sub-issues**. The fork is one API check — native sub-issues, **not**
checklists or titles. `gh` has no sub-issue subcommand; use the REST API, keyed
by the child's numeric REST **id** (the `.id` field, *not* its issue number),
passed as a typed integer (`-F`, never `-f` — a string `sub_issue_id` returns
`Invalid request`):

```
# list:   gh api repos/<owner>/<repo>/issues/<N>/sub_issues
# add:    gh api --method POST   repos/<owner>/<repo>/issues/<omnibus>/sub_issues \
#           -F sub_issue_id=$(gh api repos/<owner>/<repo>/issues/<child> --jq .id)
# remove: gh api --method DELETE repos/<owner>/<repo>/issues/<omnibus>/sub_issue \
#           -F sub_issue_id=<id>
```

Non-empty `sub_issues` → **omnibus path** (§3). Empty → **player path** (§2).
Derive a kebab slug from the issue title for worktree names.

## The plan gate
If `--plan` is set, **or** the director judges the issue underspecified or risky:
re-read the issue and its comments (already fetched in §1) and, for an omnibus,
every sub-issue; flesh out a
terse spec; present the plan — files/approach/trade-offs for a leaf, the **wave
DAG and the list of sub-issues that will fan out** for an omnibus — and **wait
for the user's OK**. Trivial, unambiguous leaves may skip the gate — say so and
proceed.

> **Before fanning out, stop and ask.** A fan-out dispatches N players in
> parallel against the tree — the costliest, least-reversible move in the skill.
> Pause and ask for guidance *liberally*; proceed only when the wave plan is
> genuinely unambiguous. When in any doubt, post the wave plan and wait.
> `--plan` makes this gate mandatory — but even without it, default to pausing here.

## 2. Player path (leaf issue)
1. Create a sibling worktree off the target (`worktree_pattern`, never nested).
2. Dispatch **one player** with: the issue + fleshed-out spec, the target branch,
   `guardrails`, and the resolved tier (Agent `model` param).
3. The player runs its loop (§The players) → pushes its branch → opens the
   **issue → target** PR.
4. On the player's branch, in parallel: the **simplicity critic** and the
   **correctness critic** review it (§The critics). Then the triage loop
   (§Triage). (No cross-cutting pass — a leaf is a set of one.)
5. **Docs sweep** (§Docs sweep) on the player's branch.
6. **Stop.** The PR is the deliverable; the human merges. (Target is `main` in
   the common case → no promotion. Target is an omnibus branch → that omnibus's
   promotion waits until it's shipped *as* an omnibus.)

## 3. Omnibus path (issue with sub-issues)
1. Ensure the **omnibus branch** exists (`branch_prefix` stem, off `base_branch`).
   Creating a new long-lived branch is a remote mutation — confirm, then
   `git branch <omnibus-branch> origin/<base_branch> && git push -u origin
   <omnibus-branch>`; never create it silently.
2. Compute the **wave plan** from the sub-issues — topological by declared
   dependencies, then by file-overlap so a wave never has two players colliding.
   Post the wave plan (and the resolved player tier) as an omnibus comment, and
   **seed the progress checklist** in the omnibus body (see *Live progress*) so
   landing status is visible on GitHub from the start.
3. **Pause before fan-out** (see the gate above), then for each wave:
   dispatch its sub-issues to **players in parallel**, each in its own sibling
   worktree off the **current** omnibus HEAD. Each player lands its sub-issue and
   opens a **"Part of #<omnibus>"** sub-PR onto the omnibus branch (no `Closes`).
4. **Merge-train** (serial, never N-way): for each sub-PR, the **simplicity
   critic** reviews its diff before integrating; apply accepted fixes to the
   sub-PR branch. Then integrate and run the integration `test_cmd` on the
   post-merge omnibus branch — green + clean → `gh pr merge`, then **tick this
   sub-issue's checklist line** `[ ]→[x]` with its sub-PR number (see *Live
   progress*); conflict or red → **park** (leave the sub-PR open, record it). No
   unattended conflict resolution.
5. Once every wave has landed and the players have exeunt, the **correctness
   critic** and the **cross-cutting simplicity critic** review the assembled
   omnibus, in parallel (§The critics); run the triage loop (§Triage).
6. **Docs sweep** (§Docs sweep) on the omnibus branch.
7. **Promote:** reconcile the native sub-issue links against the final landed set
   (add any tracked-but-unlinked, remove a stale link), then open the **omnibus →
   `base_branch`** PR enumerating `Closes #<N>` for the bundle, with an at-a-glance
   triage report (§Report). **Stop.** A human merges — that one merge auto-closes every
   sub-issue and the omnibus.

### Live progress
A comment-delimited block in the omnibus body tracks **branch-landing** status,
giving live GitHub-visible progress as waves land:

```
<!-- omnibus-checklist:start -->
- [ ] #<N> — <short label>
<!-- omnibus-checklist:end -->
```

Seed it at fan-out (§3.2) with one `- [ ]` line per sub-issue. The moment a
sub-PR lands in the merge-train (§3.4), flip its line to
`- [x] #<N> — <short label> (#<sub-PR>)`. **Edit only between the markers, by
anchored match** — never free-form rewrite the curated body. The markers let
`/ship` find and edit the block without disturbing the surrounding prose; it
complements (does not replace) the one-shot wave-plan comment.

## The players
A player is an isolated subagent (Agent tool, `isolation: worktree`) in its own
sibling worktree. It runs under your login, so it needs no separate auth and
inherits the session's permission posture. Its loop:

1. Implement the issue in logical units.
2. For **every** code-producing commit, in order:
   a. `test_cmd` — must pass. Skipped when `--skip-tests`, or the diff is wholly
      within `auto_skip_globs`, or `test_cmd` is `""`. Note once when skipped.
   b. **Self-tidy** — reread your own diff and remove the obvious: dead code,
      needless abstraction, a thing you just wrote that the codebase already had.
      (The simplicity critic reviews independently once you open your PR — just
      don't leave obvious mess.)
   c. Commit (in the worktree; never on the base branch).
3. If `decisions_dir` is set, record any decision as its own additive **file**,
   `<decisions_dir>/GH-<issue:0000>-<slug>-0001.md` (issue number four-zero-padded;
   the trailing index climbs past `0001` only for a second decision on the same
   issue) — the file only, **not** the `decisions_index` line (the director
   consolidates the index in the docs sweep).
4. If `--with-playwright` is active **and** the diff touches
   `[playwright].frontend_glob`, launch `dev_server` and verify in a real browser;
   attach before/after screenshots. If the Playwright MCP isn't connected, say so.
5. Push; open the PR (issue→target for a leaf, "Part of #<omnibus>" for a wave).
6. A blocked or genuinely-unsure player — or one that would have to cross
   `guardrails` — **parks with a written question** and never guesses. The plan
   gate is what keeps parks rare.

## The critics
The director spools the critics as subagents, independent of the authors. Their
findings are **advisory** — the director triages them (§Triage); critics never
commit. Two of them:

### The correctness critic
Hunts bugs, especially at integration seams. Risk-ranked, plain-language
findings. Reviews the leaf branch, or the assembled omnibus.

### The simplicity critic
Reviews for needless complexity, at two moments:

- **Per-unit** (cheap) — reviews each leaf branch, and each sub-PR as it lands in
  the merge-train, before it's integrated.
- **Cross-cutting** (strong) — after assembly, reviews the whole omnibus for what
  only a wide view shows: **cross-issue and cross-codebase duplication** (two
  sub-issues built the same helper; a player reinvented an existing util) and
  seam-level over-engineering. It hunts overlaps, not the per-diff nits the
  per-unit pass already covered.

Its brief, over the given diff:

> You are a refactoring specialist with one conviction: the best code is the
> least code that clearly expresses intent. Every line must earn its place — you
> *remove* complexity, you do not add cleverness. Review in three passes, macro
> to micro, and report **only what you're confident about** — never taste, style,
> or bikeshedding, and **never bugs** (the correctness critic owns those).
>
> 1. **Architectural.** Does this component need to exist, or can its
>    responsibility be absorbed elsewhere? Unnecessary layers of indirection,
>    interfaces, or patterns used for their own sake? Is the data flow as direct
>    as it could be? Ask: "if I built this from scratch today, would I structure
>    it this way?"
> 2. **File-level.** Is the public surface (exports, methods) as small as it can
>    be? Functions/classes doing too much — or over-split? Pass-through
>    wrappers/adapters that only forward? Dead paths, unused params, vestigial
>    logic? Nesting that early returns / guard clauses would flatten?
> 3. **Line-level.** Duplication — extract it, DRY is non-negotiable. Needlessly
>    complex conditionals, verbose non-idiomatic patterns, single-use variables,
>    catch-just-to-rethrow, comments that merely restate the code.
>
> Hunt specifically for: **premature abstraction** (serves one caller),
> **speculative generality / gold-plating** (built for hypothetical futures),
> **copy-paste duplication**, **unnecessary mutable state**, and
> **abstraction-astronautics** (factory-factories, excessive DI).
>
> **Do not add complexity in the name of "best practices"** — no class where a
> function works, no helper extracted for a single caller, no nesting where flat
> reads fine. **Preserve behavior** — refactoring is same behavior, better
> structure. Don't DRY tests at the expense of readability. When two approaches
> tie, choose the one with fewer concepts to understand. If a touched file is
> `CLAUDE.md` / `SKILL.md`, apply the same lens to its *prose* — but **tighten and
> dedupe wording only; never drop an actual rule or constraint** just because it
> could be shortened; when in doubt, flag the line as a question rather than cut it.
>
> Output: per finding, 1–2 sentences on what you found and why it's a problem,
> then the simpler form.

Either critic may be **skipped for a trivial or docs-only diff** (same spirit as
the test auto-skip) — say so when you do. A leaf has no cross-issue dimension, so
its per-unit simplicity pass plus the correctness critic are the whole review.

## Triage (the loop)
Union the critics' findings and dedupe. Then, up to `max_critic_passes`:

- **small / integration-seam fix** → the director commits it directly (an
  "integrator commit"), then re-runs the suite.
- **substantial / in-issue rework** → a **fresh player** off the current HEAD
  carrying a rework objective + a new sub-PR. Re-dispatching for a one-liner is
  wrong-footed (HEAD has moved under the original player) — only "the whole
  approach is off" goes back to a player.

Re-run the relevant critic after fixes. Clean, or the cap is hit → proceed.
Anything unresolved at the cap → **park** it into the report; never force a merge.

## Docs sweep
Once the work has landed and survived triage — before a leaf's PR is the
deliverable, before an omnibus is promoted — the director checks that the
change's documentation kept pace:

- **README / `architecture_doc` / the skill's `SKILL.md`** — does the change need
  a doc update (a new flag, changed behavior, a removed option)? Bring the
  current-state docs in line. Skip `architecture_doc` if it's `""`.
- **Decision log** — consolidate `decisions_index` in one newest-first pass over
  the decision files the players wrote (§The players). Decisions are **additive**:
  never fold one into `architecture_doc`'s current-state prose, never edit or
  prune an existing decision; a decision that overrides an earlier one says
  "supersedes GH-NNNN" in its own file. If `decisions_dir` / `decisions_index` is
  `""`, the project keeps no log — skip, no nagging.

Sweep edits that stay within the documentation set keep the diff docs-only, so the
test gate auto-skips (`auto_skip_globs`); a sweep that touches source re-runs it.

## Report (promotion PR body)
The omnibus→`base_branch` PR body opens with an **at-a-glance triage** — a
red/yellow/green reading that answers a reviewer's only first question, *where do
my eyes need to go?*, before they read a word. (Leaf issue→target PRs don't get
this; it's a promotion-only summary over the whole bundle.)

- **🔴 CHECK** — a human should look before merging: parked items (§Triage) with
  their written questions, critic findings that survived the cap unresolved, any
  judgment call the director made that warrants a second set of eyes. Keep red
  rare and meaningful — its value is that it's short.
- **🟡 FYI** — worth knowing, no action needed: decisions/tradeoffs taken, a
  manual conflict park later resolved, and **which tier ran** (cheap players,
  `--strong`, `--thrifty`) — a reviewer weighs a cheap-tier diff a touch more
  skeptically. Not a dumping ground; if it won't inform a merge call, cut it.
- **🟢 OK** — one line, not a list: what was verified and held (e.g. `tests green
  · both critics clean · 4/4 sub-PRs landed clean`). Break out an individual green
  item only when it's something that plausibly could have failed and didn't (a
  risky integration seam that held).

Below the triage, the plain-language detail: the `Closes #<N>` bundle that landed
and the full text of any parked questions. Always pair the light with its word
(🔴 CHECK / 🟡 FYI / 🟢 OK) so it survives where color alone doesn't.

## Cleanup
After the user confirms the merge, remove the worktree(s) you created and
fast-forward the base. On restart, adopt or clean **only** your own orphaned
worktrees — never one you didn't create.
