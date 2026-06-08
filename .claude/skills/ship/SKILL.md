---
name: ship
description: >-
  Implement a GitHub issue end-to-end the way this repo expects: sync the base
  branch, sibling worktree, pre-commit gates, conflict reconciliation, and a PR
  that closes the issue. Invoke as `/ship #<N>` (or `/ship <N>`); add `onto
  #<omnibus>` to ship onto an omnibus branch instead of `main`, or pass an omnibus
  issue alone to open its promotion PR to `main`. Use whenever asked to ship,
  implement, or do a numbered issue in bartleby.
---

# ship — issue → tested PR

Argument is one GitHub issue number (`#118`, `118`), optionally followed by any
of three opt-in tokens — `onto #<omnibus>`, `with-playwright`, and `skip-tests`
(e.g. `/ship #134 with-playwright`, `/ship #170 onto #169 skip-tests`, in any
order) — see below. Run the steps in order. Two hard stops are marked **PAUSE**:
do not pass them without the user's OK. One exception: if the lone issue is itself
an *omnibus issue* (no `onto`), `/ship` runs **omnibus-promotion mode** — opening
the omnibus → `main` PR instead of implementing anything (see below).

Throughout, **base** is the integration branch this issue targets: `main` by
default, or the omnibus branch when `onto #<omnibus>` is given (see below).
Wherever a step says `origin/<base>`, read `origin/main` in the normal case.

Repo specifics: tests are `uv run pytest`; the main checkout is
`/Users/johnwest/Code/spot/bartleby`; worktrees are **siblings** of it. A
PreToolUse hook (`guard-main-write.sh`) blocks commits/pushes on `main` — treat
a block as a signal you're on the wrong branch, not an error to route around.

**Optional `onto #<omnibus>`.** Ship onto an *omnibus/integration branch* instead
of `main`, for landing several sub-issues of a bundle (e.g. a release) before it
reaches `main` as one unit. The token is the **omnibus issue number**, not a
branch name. When present:
- **Resolve the branch.** Read the omnibus issue (`gh issue view <omnibus>`) and
  derive its branch from the title's leading version: a title `vX.Y.Z — …` yields
  `omnibus/vX.Y.Z` (bartleby omnibus issues are titled this way). If the title
  has no parseable leading `vX.Y.Z`, **refuse** — "#<omnibus> isn't titled `vX.Y.Z
  — …`; can't derive an omnibus branch" — don't invent a name.
- **Ensure the branch exists.** `git ls-remote --exit-code --heads origin
  <omnibus-branch>`. If it exists, use it. If not, this is the bundle's first
  ship: **confirm with the user**, then create it from `origin/main` and push it
  (`git branch <omnibus-branch> origin/main && git push -u origin
  <omnibus-branch>`) so the remote ref exists for the worktree base and later
  sub-ships. Report whichever happened — pushing a new long-lived branch is a
  remote mutation, never create it silently.
- **`base` becomes `<omnibus-branch>`** for every step below: sync, collision
  scan, worktree start-point, the pytest-skip checks (both paths), reconcile, and the PR base,
  diff-stat, and cleanup all retarget from `origin/main` to
  `origin/<omnibus-branch>`.
- **Closure differs.** GitHub only auto-closes from the default branch, so a
  sub-PR merged into the omnibus branch closes nothing. The sub-PR body says
  **"Part of #<omnibus>"** with **no `Closes` keyword**; sub-issues close when the
  omnibus → main PR (which enumerates `Closes #<N>` for the bundle) merges. Do not
  put `Closes #<N>` on a sub-PR in this mode.
- **Keep the omnibus issue's tracking current.** Promotion mode draws its `Closes
  #<N>` manifest from the omnibus issue's hand-curated sub-issue checklist, which
  drifts unless each sub-ship updates it. So when the sub-PR is opened (step 11),
  edit the omnibus issue (`#<omnibus>`) to reflect this landing: tick this issue's
  box `[ ]→[x]`, annotate it with the sub-PR number and target in the existing
  style (`— … (#NNN)`), and flip any dependency/order marker the omnibus tracks.
  **Edit only the one checklist line, by anchored match** — never free-form rewrite
  the hand-curated body — and show the proposed issue diff at the step-11 PAUSE next
  to the PR draft. If the omnibus issue has no checklist line referencing `#<N>`,
  **report it and leave the body untouched** rather than invent tracking. (Ticking
  on sub-PR-open is safe — promotion reconciles `Closes` against actually-merged
  PRs.)
- The guard hook still protects `main` only; the omnibus branch is not
  hook-protected. Composes with `with-playwright` and `skip-tests`.

**Omnibus-promotion mode (`/ship #<omnibus>`, no `onto`).** When the lone argument
is itself an *omnibus issue* — its title parses as `vX.Y.Z — …` **and** an
`omnibus/vX.Y.Z` branch exists with commits ahead of `origin/main` — `/ship`
implements nothing (the bundle already landed on that branch). It opens the
**omnibus → `main` PR** that promotes the whole bundle. If the title looks
omnibus-shaped but the branch is missing or not ahead of `main`, **report and
stop** — don't guess. The flow is reduced:
- **Steps 1–2 only.** Preconditions (clean main) and sync (`git fetch origin`, ff
  `main`). **No worktree, no code, no per-commit gates, no reconcile** — steps
  3–10 are N/A.
- **Build the `Closes` enumeration.** Take the bundle's issue numbers from the
  omnibus issue's sub-issue checklist, then **reconcile against the sub-PRs
  actually merged into the branch**
  (`gh pr list --base omnibus/vX.Y.Z --state merged --json number,title`, cross-
  checked with `git log origin/main..origin/omnibus/vX.Y.Z`). Emit `Closes #<N>`
  for every sub-issue whose work is on the branch. If the checklist and the
  merged-PR set disagree, **report the discrepancy** and let the user reconcile — a
  hand-edit slip must not silently add or drop a `Closes`.
- **PAUSE — PR (step 11).** Same PAUSE contract as step 11, but the body is the
  `Closes #<N>` list + a one-line bundle summary and the diff is `git diff --stat
  origin/main...origin/omnibus/vX.Y.Z`. Then `gh pr create --base main --head
  omnibus/vX.Y.Z`. **Do not merge** — the human merge auto-closes the whole bundle
  (every sub-issue and the omnibus issue).
- **Cleanup (step 12).** After the user confirms the merge, there's no worktree to
  remove; just `git pull --ff-only origin main` in the main checkout. Cutting the
  release is the separate, later `/release` act on `main` — promotion PR → human
  merge → `/release`, never folded into this step.

**Optional `with-playwright`.** The argument may carry a `with-playwright` token
after the issue number (only that exact token). It turns on a visual-verification
loop, and it fires **only if this issue's diff touches `bartleby/web/`** — for a
backend-only change, ignore it and ship normally. When it's active:
- Drive a real browser with the **Playwright MCP** tools. Launch the dev server
  with `npm run dev` in the worktree (vite on `127.0.0.1:5173`) and stop it when
  you're done; or invoke the built-in `run` skill to launch the app.
- Screenshot each affected route as a baseline before you change it, then
  re-screenshot and compare after each logical unit (step 8) and once more before
  the PR (step 11). Read the screenshots and iterate on what looks wrong.
- If the Playwright MCP isn't connected, say so and ask how to proceed — don't
  silently skip the loop.

Don't edit the built-in `verify`/`run` skills; this flag lives entirely here.

**Skipping the pytest gates.** The `uv run pytest` runs in steps 8, 9, and 10 are
omitted when **either** of these holds — evaluate both against
`git diff --name-only origin/<base>...HEAD`:

1. **Docs-only (automatic, no token).** The diff touches *only* documentation —
   every changed path is a `*.md` file, `LICENSE`, or under `docs/`. Then pytest
   can't be affected, so the gates skip on their own; no `skip-tests` token needed.
2. **`skip-tests` token (opt-in).** The argument carries the `skip-tests` token
   (only that exact token) **and** the diff touches no test-affecting source — no
   `*.py` and no `pyproject.toml`. The token is a convenience for work that's
   test-irrelevant but falls outside the path-1 docs set: a frontend-only change
   under `bartleby/web/` (Svelte/CSS/vite config — that tree holds no Python), a
   top-level `.sh` script, a `.txt` asset. It is **not** a way to land untested
   code. If the token is present but the diff *does* touch `*.py` or
   `pyproject.toml`, **ignore the token and run the tests anyway**, and say why
   ("`skip-tests` requested but the diff changes `bartleby/commands/ready.py` —
   running tests anyway"). One accepted gap: a *structural* `bartleby/web/` change
   (move `src/`, drop `package.json`) can still fail the Python suite via
   `tests/test_serve.py`, which asserts the packaged UI layout — `skip-tests` won't
   catch that, so don't pair the token with a web restructure.

Re-check **both** paths at **each** gate, not just once: a docs PR that grows a
code change mid-stream must start running tests from that point, and a diff that
narrows back to docs-only resumes path-1 skipping (token or not). When tests are genuinely skipped,
**say so** in the step-11 PR summary and the final report, naming the path —
"Tests skipped — docs-only diff" (path 1) or "Tests skipped — `skip-tests`, no
`.py`/`pyproject.toml` touched" (path 2) — so a skipped suite never reads as a green one. The
simplify-refactor pass (step 8) still runs regardless.

## 1. Preconditions
- `git -C <main-checkout> status --porcelain` must be empty. If main has
  uncommitted changes, stop and report — don't proceed on a dirty tree.
- Confirm `gh auth status` is good.

## 2. Sync the base
From the main checkout: `git fetch origin`. In the normal case also fast-forward
main: `git pull --ff-only origin main`. The worktree (step 5) is always based on
the freshly-fetched `origin/<base>`, so onto-mode needs only the fetch — don't
check out the omnibus branch in the main checkout (creating it, if it didn't
exist, already happened during `onto` flag-resolution above — that's a separate
`git branch`/`push`, not a checkout).

## 3. Read the issue
`gh issue view <N>`. Derive a kebab slug from the title → branch
`issue/<N>-<slug>`, worktree `../bartleby-issue-<N>-<slug>`.

## 4. Collision scan (do this before creating the worktree)
Surface overlapping in-flight work so a conflict is known up front:
- `git worktree list` and `gh pr list --state open` — note other active
  branches/PRs.
- For each, compare its changed files (`git diff --name-only origin/<base>...<branch>`
  / `gh pr diff <n> --name-only`) against the files this issue will likely touch.
- If there's overlap, **report it** ("PR #128 also edits `scan.py`") and let the
  user decide whether to proceed, reorder, or coordinate. Don't silently barrel in.

## 5. Create the worktree
`git worktree add ../bartleby-issue-<N>-<slug> -b issue/<N>-<slug> origin/<base>`
— the explicit `origin/<base>` start-point bases the branch on the integration
target (`main` normally, the omnibus branch under `onto`). Never `git checkout -b`
inside the main checkout; never nest the worktree in the repo.

## 6. Flesh out a terse issue
If the body is empty/thin, once the approach is settled write it back with
`gh issue edit <N>` (problem + approach + scope + follow-ups) **before** coding.
File deliberately-scoped-out work as its own issue and reference it.

## 7. PAUSE — plan
For any non-trivial issue, present the implementation plan (files, approach,
trade-offs) and **wait for approval** before writing code. Trivial one-liners may
skip this — say so and proceed.

## 8. Implement + per-commit gates
Work in logical units. If `with-playwright` is active, bracket each web-touching
unit with before/after screenshots (see the flag note above). For **every**
code-producing commit, in this exact order:
1. `uv run pytest` — must pass. (Skipped when either skip path holds — a docs-only
   diff or `skip-tests` with no `.py`/`pyproject.toml` touched; see the "Skipping
   the pytest gates" note above and re-check both paths here.)
2. Run the `simplify-refactor` agent against the just-touched files.
3. Apply the suggestions you agree with; push back on the rest.
4. Re-run `uv run pytest` — must still pass. (Same skip condition as 1.)
5. Commit (in the worktree; the hook enforces you're off `main`).

## 9. Docs sweep
Check README / ARCHITECTURE.md / skill `SKILL.md` for updates the change
requires (new flags, changed behavior, a decision-log entry). If you change docs,
re-run the step-8 gates before continuing (the skip condition from step 8 applies
to the pytest run here too — note a docs-sweep edit that stays within the
documentation set keeps a diff docs-only, so path 1 still skips).

## 10. Reconcile before the PR
Bring the branch up to date so conflicts surface here, not in the PR:
`git fetch origin && git merge origin/<base>` (or rebase). Resolve any conflicts,
then run the **full** suite (`uv run pytest`) again — unless either skip path
still holds (docs-only diff, or `skip-tests` with no `.py`/`pyproject.toml` touched). Re-run the same
`origin/<base>...HEAD` check from the flag note; because it's three-dot (the diff
since the merge-base), merging `origin/<base>` in doesn't change what it sees.

## 11. PAUSE — PR
Show the user the **PR body draft + a final diff summary** (`git diff --stat
origin/<base>...HEAD`) and wait for their OK. Then push and `gh pr create`.
**Do not merge** — the user merges.
- Normal case: target `main` (the default) with a `Closes #<N>` line.
- Under `onto`: pass `--base <omnibus-branch>`, and write **"Part of #<omnibus>"**
  with **no `Closes`** (see the flag note) — the sub-issue closes at the omnibus →
  main merge, not here. Also show and apply the omnibus-issue tracking edit here
  (see *Keep the omnibus issue's tracking current*).

## 12. Cleanup (only after the user confirms the merge)
From the main checkout: `git worktree remove ../bartleby-issue-<N>-<slug>`,
`git branch -D issue/<N>-<slug>` (squash-merge leaves it "unmerged"; `-D` is
expected). Then refresh: normally `git pull --ff-only origin main`; under `onto`
this issue's sub-PR merged into the omnibus branch (not main — the omnibus → main
merge is a separate, later act), so just `git fetch origin` to update
`origin/<base>`. **Touch only this issue's worktree/branch** — never remove
others' even if they look stale; flag them instead.
