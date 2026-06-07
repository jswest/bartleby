---
name: ship
description: >-
  Implement a GitHub issue end-to-end the way this repo expects: sync main,
  sibling worktree, pre-commit gates, conflict reconciliation, and a PR that
  closes the issue. Invoke as `/ship #<N>` (or `/ship <N>`). Use whenever asked
  to ship, implement, or do a numbered issue in bartleby.
---

# ship — issue → tested PR

Argument is one GitHub issue number (`#118`, `118`), optionally followed by the
`with-playwright` token (e.g. `/ship #134 with-playwright`) — see below. Run the
steps in order. Two hard stops are marked **PAUSE**: do not pass them without the
user's OK.

Repo specifics: tests are `uv run pytest`; the main checkout is
`/Users/johnwest/Code/spot/bartleby`; worktrees are **siblings** of it. A
PreToolUse hook (`guard-main-write.sh`) blocks commits/pushes on `main` — treat
a block as a signal you're on the wrong branch, not an error to route around.

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

## 1. Preconditions
- `git -C <main-checkout> status --porcelain` must be empty. If main has
  uncommitted changes, stop and report — don't proceed on a dirty tree.
- Confirm `gh auth status` is good.

## 2. Sync main
From the main checkout: `git fetch origin && git pull --ff-only origin main`.

## 3. Read the issue
`gh issue view <N>`. Derive a kebab slug from the title → branch
`issue/<N>-<slug>`, worktree `../bartleby-issue-<N>-<slug>`.

## 4. Collision scan (do this before creating the worktree)
Surface overlapping in-flight work so a conflict is known up front:
- `git worktree list` and `gh pr list --state open` — note other active
  branches/PRs.
- For each, compare its changed files (`git diff --name-only origin/main...<branch>`
  / `gh pr diff <n> --name-only`) against the files this issue will likely touch.
- If there's overlap, **report it** ("PR #128 also edits `scan.py`") and let the
  user decide whether to proceed, reorder, or coordinate. Don't silently barrel in.

## 5. Create the worktree
`git worktree add ../bartleby-issue-<N>-<slug> -b issue/<N>-<slug>`.
Never `git checkout -b` inside the main checkout; never nest the worktree in the repo.

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
1. `uv run pytest` — must pass.
2. Run the `simplify-refactor` agent against the just-touched files.
3. Apply the suggestions you agree with; push back on the rest.
4. Re-run `uv run pytest` — must still pass.
5. Commit (in the worktree; the hook enforces you're off `main`).

## 9. Docs sweep
Check README / ARCHITECTURE.md / skill `SKILL.md` for updates the change
requires (new flags, changed behavior, a decision-log entry). If you change docs,
re-run the step-8 gates before continuing.

## 10. Reconcile before the PR
Bring the branch up to date so conflicts surface here, not in the PR:
`git fetch origin && git merge origin/main` (or rebase). Resolve any conflicts,
then run the **full** suite (`uv run pytest`) again.

## 11. PAUSE — PR
Show the user the **PR body draft + a final diff summary** (`git diff --stat
origin/main...HEAD`) and wait for their OK. Then push and
`gh pr create` with a `Closes #<N>` line. **Do not merge** — the user merges.

## 12. Cleanup (only after the user confirms the merge)
From the main checkout: `git worktree remove ../bartleby-issue-<N>-<slug>`,
`git branch -D issue/<N>-<slug>` (squash-merge leaves it "unmerged"; `-D` is
expected), then `git pull --ff-only origin main`. **Touch only this issue's
worktree/branch** — never remove others' even if they look stale; flag them instead.
