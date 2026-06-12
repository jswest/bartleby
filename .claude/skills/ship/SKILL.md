---
name: ship
description: >-
  Implement a GitHub issue end-to-end the way a repo expects: sync the base
  branch, sibling worktree, per-commit gates, conflict reconciliation, and a PR
  that closes the issue. Invoke as `/ship #<N>` (or `/ship <N>`); add `onto
  #<omnibus>` to ship onto an omnibus branch instead of the base, or pass an
  `Omnibus:`-titled issue alone to open its promotion PR. Portable and
  config-driven via `.claude/ship.toml`. Use whenever asked to ship, implement,
  or do a numbered issue.
---

# ship — issue → tested PR

Argument is one GitHub issue number (`#118`, `118`), optionally followed by any
of three opt-in tokens — `onto #<omnibus>`, `with-playwright`, and `skip-tests`
(e.g. `/ship #134 with-playwright`, `/ship #170 onto #169 skip-tests`, in any
order) — see below. Run the steps in order. Two hard stops are marked **PAUSE**:
do not pass them without the user's OK. One exception: if the lone issue is itself
an *omnibus issue* (no `onto`), `/ship` runs **omnibus-promotion mode** — opening
the omnibus → base PR instead of implementing anything (see below).

Throughout, **base** is the integration branch this issue targets: `base_branch`
(the repo's default branch, from config) by default, or the omnibus branch when
`onto #<omnibus>` is given (see below). Wherever a step says `origin/<base>`,
read `origin/<base_branch>` in the normal case.

This skill is **portable**: one source lives in the drawer at
`~/Code/workflows/skills/ship/` and is *pressed* by `signet` into each repo's
`.claude/skills/ship/`. Everything repo-specific — the test command, the
quality-gate agent, the worktree scheme, the doc paths, the optional
visual-verification and live-data redlines — lives in a per-repo
`.claude/ship.toml` that the skill **detects once and persists**. The prose below
never hard-codes a repo's specifics; it reads them from config.

## Per-project config — `.claude/ship.toml`

Each repo's specifics live in `.claude/ship.toml`. First run **detects and
proposes**; on approval it **persists**; later runs **read it directly**. The
skill never reaches for a repo's specifics from anywhere else. The config is a
sibling of the pressed skill dir, so `signet` never touches it.

```toml
# .claude/ship.toml — written by `/ship config`, editable by hand. Commit it.
base_branch      = "main"                          # default branch / promotion target
branch_prefix    = "omnibus"                       # omnibus branch stem (shared with ultraship)
version_scheme   = ""                              # "" = {prefix}-{slug};
                                                   #   "leading-v" = prefer {prefix}/vX.Y.Z when a
                                                   #   version is parseable from the omnibus title.
worktree_pattern = "../{repo}-issue-{n}-{slug}"    # sibling, never nested

test_cmd          = "uv run pytest"                # "" = no suite (rare; warn)
test_source_globs = ["*.py", "pyproject.toml"]     # a diff touching any of these is "test-affecting"
docs_only_globs   = ["*.md", "LICENSE*", "docs/**"] # a diff wholly within these auto-skips the suite
skip_tests_token  = true                           # honor a `skip-tests` opt-in
gate_agent        = "simplify-refactor"            # per-commit quality pass; "" = skip that gate + note

architecture_doc  = "ARCHITECTURE.md"              # current-state doc swept in step 9 ("" = none)
decisions_dir     = "docs/decisions"               # "" = project keeps no decision log
decisions_index   = "docs/decisions/README.md"     # "" = no newest-first index to maintain

guard_hook        = ".claude/hooks/guard-main-write.sh"  # detected; informational only

# Free text injected VERBATIM into the step-8 implementation guidance — a repo's
# "never touch live X" redline (analog of ultraship's player_guardrails). "" = none.
#   e.g. "Tests run on checked-in fixtures only. Probe the live site once, by hand,
#         then turn it into a fixture — never a network-touching test."
live_data_note = ""

# Optional feature block — present ⇒ `with-playwright` is available; absent ⇒
# the token is unknown and ignored. (Keep this table LAST: in TOML every bare key
# after a [table] header belongs to that table.)
[playwright]
frontend_glob = "web/**"                           # with-playwright fires only if the diff touches this
dev_server    = "npm run dev"                      # how to launch the dev server in the worktree
url           = "http://127.0.0.1:5173"            # where to point the browser
```

### Detection heuristics (first run / `/ship config` / `--reconfigure`)

Auto-derive every field; ask only where a value is genuinely ambiguous. Never
silently guess a value that would mislead — show everything for approval.

- `base_branch`: the repo's default branch
  (`gh repo view --json defaultBranchRef --jq .defaultBranchRef.name`).
- `worktree_pattern`: default `../{repo}-issue-{n}-{slug}`, `{repo}` = basename of
  the main checkout (`git rev-parse --show-toplevel`).
- `test_cmd`: `pyproject.toml` + `uv.lock` → `uv run pytest`; `package.json` with
  a `test` script → `npm test` (or `pnpm`/`yarn` per lockfile); `Cargo.toml` →
  `cargo test`; `go.mod` → `go test ./...`; a `Makefile` `test:` target → `make
  test`. Ambiguous / none → **ask**, don't guess.
- `test_source_globs`: from the same project type — Python → `["*.py",
  "pyproject.toml"]`; Node → `["*.ts", "*.js", "package.json"]`; Rust → `["*.rs",
  "Cargo.toml"]`; etc. This is the path-2 "test-affecting source" set.
- `gate_agent`: probe the agent registry for `simplify-refactor` (or the repo's
  documented quality agent); if none, leave `""` and skip that gate half (noted).
- `architecture_doc`: the repo's current-state design doc if one exists
  (`ARCHITECTURE.md`, `SPEC.md`, `DESIGN.md`); else `""`.
- `decisions_dir`/`decisions_index`: present iff the path exists; absent → `""`.
- `version_scheme`: `"leading-v"` iff the repo already versions — git tags shaped
  `vX.Y.Z`, a `__version__`/`version =` in packaging, or an omnibus title carrying
  a `vX.Y.Z`; else `""`. Never required.
- `guard_hook`: scan `.claude/settings*.json` PreToolUse hooks + `.claude/hooks/`.
- `[playwright]`: propose a block only if the repo has a frontend tree (a
  `package.json` with a dev script under a `web/`-like dir); otherwise omit it.

### First run / `/ship config`

A `/ship #<N>` in a repo with **no** `.claude/ship.toml` **routes into config
first** (it does not silently autogenerate). `/ship config` invokes it on demand;
`--reconfigure` re-runs it over an existing file. Run detection, walk the fields
in plain language each with its detected value as the default, show the assembled
`.toml`, and **write it on approval**. Done once per repo.

**Commit policy: commit it.** The `.toml` holds facts about the repo and how it
ships — base branch, test command, gate agent, doc paths, the live-data redline —
identical for everyone, so it belongs in git. The redline (`live_data_note`)
especially must be shared so a fresh clone or CI run doesn't get an empty
guardrail. The rare repo that wants it machine-local hand-adds one `.gitignore`
line.

## Signet self-check

This skill is pressed from a drawer by `signet`, which leaves a `.stamp`
(`source`, `pressed_at`, per-file `sha256`) beside the pressed files. `/ship` is
an **attended** verb, so honor the stamp at the start:

- Read `.claude/skills/ship/.stamp`, find the drawer path it names, and
  byte-compare the pressed files. **Drift → stop and prompt to re-press**
  (`signet press ship --into <repo>`) before shipping.
- **If the drawer path in `.stamp` doesn't exist** on this machine → **skip
  silently** (a clone, CI, or a teammate without the drawer is never penalized).
- A **`--no-signet-check`** flag bypasses the check.

## Optional `onto #<omnibus>`

Ship onto an *omnibus/integration branch* instead of the base, for landing
several sub-issues of a bundle (e.g. a release) before it reaches the base as one
unit. The token is the **omnibus issue number**, not a branch name. When present:

- **The omnibus tracking — two complementary layers.** `/ship` keeps an omnibus
  tracked two ways, both load-bearing and answering different questions:

  1. **Native GitHub sub-issues** — each tracked issue is linked as a *sub-issue*
     of the omnibus parent (GitHub's hierarchy / progress panel). `gh` has no
     sub-issue subcommand, so use the REST API keyed by the child's numeric REST
     **id** (the `.id` field, **not** its issue number), passed as a typed integer
     (`-F`, never `-f` — a string `sub_issue_id` returns `Invalid request`):
     ```
     # list:   gh api repos/<owner>/<repo>/issues/<omnibus>/sub_issues
     # add:    gh api --method POST   repos/<owner>/<repo>/issues/<omnibus>/sub_issues \
     #           -F sub_issue_id=$(gh api repos/<owner>/<repo>/issues/<N> --jq .id)
     # remove: gh api --method DELETE repos/<owner>/<repo>/issues/<omnibus>/sub_issue \
     #           -F sub_issue_id=<id>
     ```
     **List first and add only what's missing** (idempotent). The link set is
     **reconciled to the issues the omnibus body actually tracks at two moments**
     — when the block is *seeded* (first ship) and at *promotion* — so an issue
     deferred out of the bundle isn't linked and a stale link gets removed; a
     per-sub-ship just *ensures its own* link (it doesn't prune). This panel tracks
     issue **closure**; under `onto` a sub-issue stays open until the omnibus →
     base promotion, so the panel reads 0-closed until then.
  2. **The checklist block** — a comment-delimited block in the omnibus body that
     tracks the finer-grained **branch-landing** status:
     ```
     <!-- omnibus-checklist:start -->
     - [ ] #<N> — <short label>
     <!-- omnibus-checklist:end -->
     ```
     One line per sub-issue: seeded `- [ ] #<N> — <short label>`, becoming
     `- [x] #<N> — <short label> (#<sub-PR>)` on landing. The markers let `/ship`
     find, parse, and edit it by anchor without disturbing the surrounding
     hand-written prose — it is **not** a replacement for the narrative
     `### Sub-issues` section, which stays.

  The two are deliberately distinct: the panel answers "is the issue *closed*?",
  the block answers "has the work *landed on the branch*?". A body that predates
  the regime has neither; the steps below seed both (proposed, never silent) and
  otherwise degrade gracefully — they never fabricate tracking. This is the same
  regime `/ultraship` slots into; the title/branch conventions below match it.
- **Resolve the branch (version-optional, shared with `/ultraship`).** Read the
  omnibus issue (`gh issue view <omnibus>`). Its title **must start `Omnibus:`** —
  if it does not, **refuse** ("#<omnibus> isn't titled `Omnibus: …`; won't treat
  it as a bundle"); don't invent one. Derive the branch by this precedence:
  1. **Versioned** — when `version_scheme` is set *and* a `vX.Y.Z` token is
     parseable from the title (after the `Omnibus:` prefix — e.g.
     `Omnibus: v0.9.7 — …`), the branch is `<branch_prefix>/vX.Y.Z`.
  2. **Fallback** — otherwise `<branch_prefix>-<slug>`, slug from the title minus
     its `Omnibus:` prefix, kebab-cased and truncated. If that name already exists
     for a *different* omnibus, disambiguate with the issue number
     (`<branch_prefix>-<slug>-<omnibus>`).

  A missing version is **never** a refusal; only a missing `Omnibus:` prefix is.
- **Ensure the branch exists.** `git ls-remote --exit-code --heads origin
  <omnibus-branch>`. If it exists, use it. If not, this is the bundle's first
  ship: **confirm with the user**, then create it from `origin/<base_branch>` and
  push it (`git branch <omnibus-branch> origin/<base_branch> && git push -u origin
  <omnibus-branch>`) so the remote ref exists for the worktree base and later
  sub-ships. Report whichever happened — pushing a new long-lived branch is a
  remote mutation, never create it silently. On this first ship the omnibus
  carries neither tracking layer yet: **propose seeding both** — the
  `omnibus-checklist` markers with a `- [ ] #<N>` line per sub-issue the omnibus
  body names, and a native sub-issue link for each (the `gh api` adds above) — and
  show the proposed body diff plus the link set at the step-11 PAUSE with the PR
  draft; never edit the body or link sub-issues silently.
- **`base` becomes `<omnibus-branch>`** for every step below: sync, collision
  scan, worktree start-point, the test-skip checks (both paths), reconcile, and
  the PR base, diff-stat, and cleanup all retarget from `origin/<base_branch>` to
  `origin/<omnibus-branch>`.
- **Closure differs.** GitHub only auto-closes from the default branch, so a
  sub-PR merged into the omnibus branch closes nothing. The sub-PR body says
  **"Part of #<omnibus>"** with **no `Closes` keyword**; sub-issues close when the
  omnibus → base PR (which enumerates `Closes #<N>` for the bundle) merges. Do not
  put `Closes #<N>` on a sub-PR in this mode.
- **Keep the omnibus issue's tracking current.** Promotion mode draws its `Closes
  #<N>` manifest from the omnibus checklist block (above), which drifts unless each
  sub-ship updates it. So when the sub-PR is opened (step 11), edit the omnibus
  issue (`#<omnibus>`) to reflect this landing: inside the checklist block, match
  the line for `#<N>`, tick its box `[ ]→[x]`, annotate it with the sub-PR
  number in the existing style (`— … (#NNN)`), and flip any dependency/order
  marker that line carries. Also **ensure `#<N>` is linked as a native sub-issue**
  of the omnibus (list `sub_issues`; add if missing — idempotent) so the hierarchy
  panel stays complete. **Edit only that one block line, by
  anchored match** — never free-form rewrite the hand-curated body — and show
  **both** the proposed block-line diff **and** the pending sub-issue link
  (the API call is a side effect, not a body diff — surface it explicitly) at the
  step-11 PAUSE next to the PR draft. If the omnibus body
  has **no checklist block, or a block with no line referencing `#<N>`**, propose
  the fix at the PAUSE (add the `#<N>` line, or seed the whole block as on first
  ship); if the user declines, **report it and leave the body untouched** rather
  than invent tracking. (Ticking on sub-PR-open is safe — promotion reconciles
  `Closes` against actually-merged PRs.)
- The guard hook still protects the base branch only; the omnibus branch is not
  hook-protected. Composes with `with-playwright` and `skip-tests`.

**Omnibus-promotion mode (`/ship #<omnibus>`, no `onto`).** When the lone argument
is itself an *omnibus issue* — its title starts `Omnibus:` **and** its resolved
branch (by the version-optional precedence above) exists with commits ahead of
`origin/<base_branch>` — `/ship` implements nothing (the bundle already landed on
that branch). It opens the **omnibus → base PR** that promotes the whole bundle.
If the title is `Omnibus:`-shaped but the branch is missing or not ahead of the
base, **report and stop** — don't guess. The flow is reduced:
- **Steps 1–2 only.** Preconditions (clean base) and sync (`git fetch origin`, ff
  the base). **No worktree, no code, no per-commit gates, no reconcile** — steps
  3–10 are N/A.
- **Build the `Closes` enumeration.** Take the bundle's issue numbers from the
  omnibus checklist block — the `#<N>` set inside the `omnibus-checklist` markers
  (fall back to the prose `### Sub-issues` list only if the body predates the
  regime and has no block) — then **reconcile against the sub-PRs actually merged
  into the branch** (`gh pr list --base <omnibus-branch> --state merged --json
  number,title`, cross-checked with `git log
  origin/<base_branch>..origin/<omnibus-branch>`). Emit `Closes #<N>` for every
  sub-issue whose work is on the branch. If the checklist and the merged-PR set
  disagree, **report the discrepancy** and let the user reconcile — a hand-edit
  slip must not silently add or drop a `Closes`. This is also the second
  **sub-issue reconcile moment** (see the tracking layers above): bring the native
  sub-issue links in line with the final `#<N>` set — add any tracked issue still
  unlinked, remove a link the body no longer tracks — so the promoted bundle's
  hierarchy matches what it closes.
- **PAUSE — PR (step 11).** Same PAUSE contract as step 11, but the body is the
  `Closes #<N>` list + a one-line bundle summary and the diff is `git diff --stat
  origin/<base_branch>...origin/<omnibus-branch>`. Then `gh pr create --base
  <base_branch> --head <omnibus-branch>`. **Do not merge** — the human merge
  auto-closes the whole bundle (every sub-issue and the omnibus issue).
- **Cleanup (step 12).** After the user confirms the merge, there's no worktree to
  remove; just `git pull --ff-only origin <base_branch>` in the main checkout.
  Cutting a release is a separate, later act on the base branch (in repos with a
  `/release` skill, that PR → human merge → `/release`) — never folded into this
  step.

**Optional `with-playwright` (config-gated).** Available **only if `.claude/ship.toml`
declares a `[playwright]` block**; without it the token is unknown — say so and
ignore it. The argument may carry a `with-playwright` token after the issue number
(only that exact token). It turns on a visual-verification loop, and it fires
**only if this issue's diff touches `[playwright].frontend_glob`** — for a change
outside that tree, ignore it and ship normally. When it's active:
- Drive a real browser with the **Playwright MCP** tools. Launch the dev server
  with `[playwright].dev_server` in the worktree (serving at `[playwright].url`)
  and stop it when you're done; or invoke the built-in `run` skill to launch the
  app.
- Screenshot each affected route as a baseline before you change it, then
  re-screenshot and compare after each logical unit (step 8) and once more before
  the PR (step 11). Read the screenshots and iterate on what looks wrong.
- If the Playwright MCP isn't connected, say so and ask how to proceed — don't
  silently skip the loop.

Don't edit the built-in `verify`/`run` skills; this flag lives entirely here.

**Skipping the test gates.** The `test_cmd` runs in steps 8, 9, and 10 are
omitted when **either** of these holds — evaluate both against
`git diff --name-only origin/<base>...HEAD`:

1. **Docs-only (automatic, no token).** Every changed path matches a
   `docs_only_globs` entry (default `*.md`, `LICENSE*`, `docs/**`). Then the suite
   can't be affected, so the gates skip on their own; no `skip-tests` token needed.
2. **`skip-tests` token (opt-in).** The argument carries the `skip-tests` token
   (only that exact token; honored only when `skip_tests_token = true`) **and**
   the diff touches no test-affecting source — nothing matching
   `test_source_globs`. The token is a convenience for work that's
   test-irrelevant but falls outside the path-1 docs set: a frontend-only change,
   a top-level `.sh` script, a `.txt` asset, a CI config. It is **not** a way to
   land untested code. If the token is present but the diff *does* touch a
   `test_source_globs` path, **ignore the token and run the tests anyway**, and
   say why ("`skip-tests` requested but the diff changes a source path — running
   tests anyway"). Watch the one accepted gap: a *structural* change that a test
   asserts on (moving a packaged tree, dropping a manifest) can fail the suite
   without touching a `test_source_globs` path — `skip-tests` won't catch that, so
   don't pair the token with a structural move of an asserted-on layout.

Re-check **both** paths at **each** gate, not just once: a docs PR that grows a
code change mid-stream must start running tests from that point, and a diff that
narrows back to docs-only resumes path-1 skipping (token or not). When tests are
genuinely skipped, **say so** in the step-11 PR summary and the final report,
naming the path — "Tests skipped — docs-only diff" (path 1) or "Tests skipped —
`skip-tests`, no test-affecting source touched" (path 2) — so a skipped suite
never reads as a green one. The `gate_agent` pass (step 8) still runs regardless.
If `test_cmd` is `""` (no suite), note that once and skip the test runs throughout.

## 1. Preconditions
- `git -C <main-checkout> status --porcelain` must be empty. If the base checkout
  has uncommitted changes, stop and report — don't proceed on a dirty tree.
- Confirm `gh auth status` is good.

## 2. Sync the base
From the main checkout: `git fetch origin`. In the normal case also fast-forward
the base: `git pull --ff-only origin <base_branch>`. The worktree (step 5) is
always based on the freshly-fetched `origin/<base>`, so onto-mode needs only the
fetch — don't check out the omnibus branch in the main checkout (creating it, if
it didn't exist, already happened during `onto` flag-resolution above — that's a
separate `git branch`/`push`, not a checkout).

## 3. Read the issue
`gh issue view <N> --comments` — read the **comment thread**, not just the body.
A correction or an "actually do it this way" reply often lands in the comments
after the issue was filed; that's where the *"when unsure, stop and ask"* answer
already lives, so it must shape the implementation. Derive a kebab slug from the
title (minus an `Issue:` prefix if present) → branch `issue/<N>-<slug>`, worktree
from `worktree_pattern` (e.g. `../<repo>-issue-<N>-<slug>`).

## 4. Collision scan (do this before creating the worktree)
Surface overlapping in-flight work so a conflict is known up front:
- `git worktree list` and `gh pr list --state open` — note other active
  branches/PRs.
- For each, compare its changed files (`git diff --name-only origin/<base>...<branch>`
  / `gh pr diff <n> --name-only`) against the files this issue will likely touch.
- If there's overlap, **report it** ("PR #128 also edits `parse.py`") and let the
  user decide whether to proceed, reorder, or coordinate. Don't silently barrel in.

## 5. Create the worktree
`git worktree add <worktree> -b issue/<N>-<slug> origin/<base>` — the explicit
`origin/<base>` start-point bases the branch on the integration target
(`<base_branch>` normally, the omnibus branch under `onto`). Never `git checkout
-b` inside the main checkout; never nest the worktree in the repo.

## 6. Flesh out a terse issue
If the body is empty/thin, once the approach is settled write it back with
`gh issue edit <N>` (problem + approach + scope + follow-ups) **before** coding.
File deliberately-scoped-out work as its own issue and reference it.

## 7. PAUSE — plan
For any non-trivial issue, present the implementation plan (files, approach,
trade-offs) and **wait for approval** before writing code. Trivial one-liners may
skip this — say so and proceed.

## 8. Implement + per-commit gates
Work in logical units. If `with-playwright` is active, bracket each
frontend-touching unit with before/after screenshots (see the flag note above).
If `live_data_note` is set in config, honor it verbatim throughout this step. For
**every** code-producing commit, in this exact order:
1. `test_cmd` — must pass. (Skipped when either skip path holds — a docs-only diff
   or `skip-tests` with no test-affecting source touched; see the "Skipping the
   test gates" note above and re-check both paths here. Also skipped if `test_cmd`
   is `""`.)
2. Run the `gate_agent` against the just-touched files. (If `gate_agent` is `""`,
   skip this and note once that no quality gate ran.)
3. Apply the suggestions you agree with; push back on the rest.
4. Re-run `test_cmd` — must still pass. (Same skip condition as 1.)
5. Commit (in the worktree; the guard hook enforces you're off the base branch).

## 9. Docs sweep
Check README / `architecture_doc` / skill `SKILL.md` for updates the change
requires (new flags, changed behavior, a decision worth recording). **A decision
goes in `decisions_dir` as its own additive file**, never folded into the
`architecture_doc`'s current-state text: add
`<decisions_dir>/GH-<issue:0000>-<slug>-0001.md` — the issue number
four-zero-padded (`<index>` climbs past `0001` only when one issue yields more than
one decision), the decision written as a standalone note — plus a newest-first
line in `decisions_index`. Existing decisions are never edited or pruned; a new
call that overrides an old one says "supersedes GH-NNNN" in its own file. If
`decisions_dir` (or `decisions_index`) is `""`, the project keeps no decision log
— skip that step with no nagging. If `architecture_doc` is `""`, skip it too. If
you change docs, re-run the step-8 gates before continuing (the skip condition
from step 8 applies to the test run here too — note a docs-sweep edit that stays
within the documentation set keeps a diff docs-only, so path 1 still skips).

## 10. Reconcile before the PR
Bring the branch up to date so conflicts surface here, not in the PR:
`git fetch origin && git merge origin/<base>` (or rebase). Resolve any conflicts,
then run the **full** suite (`test_cmd`) again — unless either skip path still
holds (docs-only diff, or `skip-tests` with no test-affecting source touched).
Re-run the same `origin/<base>...HEAD` check from the flag note; because it's
three-dot (the diff since the merge-base), merging `origin/<base>` in doesn't
change what it sees.

## 11. PAUSE — PR
Show the user the **PR body draft + a final diff summary** (`git diff --stat
origin/<base>...HEAD`) and wait for their OK. Then push and `gh pr create`.
**Do not merge** — the user merges.
- Normal case: target `<base_branch>` (the default) with a `Closes #<N>` line.
- Under `onto`: pass `--base <omnibus-branch>`, and write **"Part of #<omnibus>"**
  with **no `Closes`** (see the flag note) — the sub-issue closes at the omnibus →
  base merge, not here. Also show and apply the omnibus-issue tracking edit here
  (see *Keep the omnibus issue's tracking current*).

## 12. Cleanup (only after the user confirms the merge)
From the main checkout: `git worktree remove <worktree>`, `git branch -D
issue/<N>-<slug>` (squash-merge leaves it "unmerged"; `-D` is expected). Then
refresh: normally `git pull --ff-only origin <base_branch>`; under `onto` this
issue's sub-PR merged into the omnibus branch (not the base — the omnibus → base
merge is a separate, later act), so just `git fetch origin` to update
`origin/<base>`. **Touch only this issue's worktree/branch** — never remove
others' even if they look stale; flag them instead.
