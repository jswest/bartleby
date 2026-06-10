# `/ultraship` players are isolated subagents, not headless `claude -p` processes; the `simplify-refactor` gate moves to the stage-manager (issue #335)

> Source: [#335](https://github.com/jswest/bartleby/issues/335)

Supersedes the player-mechanism half of
[GH-0244](GH-0244-ultraship-authority-boundary-0001.md) (its part 2, "players run
unattended with pre-granted permissions"). GH-0244's part 1 — the stage-manager
merges sub-issue → omnibus autonomously, human still merges omnibus → `main`,
guard hook untouched — **stands unchanged**.

**A player is an isolated subagent, not a headless `claude -p` process.** The
stage-manager dispatches each sub-issue to a subagent (the Agent tool with
`isolation: worktree`) running in its own sibling worktree off the current omnibus
HEAD. GH-0244 specified a headless `claude -p` process and accepted the exposure
of pre-granting it broad command access (or `--dangerously-skip-permissions`) so
it wouldn't hang on a 1 a.m. permission prompt. The subagent mechanism removes
that exposure rather than accepting it:

- **No separate auth or permission grant.** A subagent runs inside the
  stage-manager's own session under the user's login, inheriting the session's
  permission posture. There is no spawned CLI process to pre-authorize, no
  API-key-strip / own-login care to get right (the `claude -p` ToS hygiene of
  [GH-0295](GH-0295-anthropic-cc-judge-0001.md)), and no
  `--dangerously-skip-permissions` switch — so the "hangs at 1 a.m. with nobody to
  approve" failure mode GH-0244 designed around cannot occur. The choice is
  ToS-clean by construction: same session, same login, no second process.
- **It matches observed practice.** A stage-manager exercising the harness already
  dispatches players as subagents rather than nesting `claude -p` inside a
  background job.

**The trade-off, and where the `simplify-refactor` gate goes.** A subagent cannot
spawn another agent. So the per-player gate's `simplify-refactor` step — which is
an *agent*, not a shell command — cannot run inside a subagent player. The gate
therefore splits:

- **Players keep the `uv run pytest` half** (they have shell access) →
  `pytest` → commit, with `/ship`'s pytest-skip rules.
- **The stage-manager runs `simplify-refactor`** over each sub-PR's diff during
  the serialized merge train (run step 4), pushing accepted fixes onto the sub-PR
  branch before merging. The stage-manager is the one role that *can* spawn agents
  (it is the orchestrating session, not a subagent), and the merge train is
  already a single-serial-writer seam — so the gate runs in integration context
  with no nesting and no parallel-writer hazard. This is strictly better than the
  in-player gate degrading silently to a self-review on a refactor-heavy bundle.

**Why this is not a softening.** The full gate sequence still runs for every
sub-issue — `pytest` in the player, `simplify-refactor` at the stage-manager —
just split across the two roles that can each execute their half. The
park-with-a-written-question escape valve, the recoverable-branch / serialized /
never-force-pushed merge discipline, the GitHub-as-state model, and the
human-merges-`main` boundary are all unchanged from GH-0244.

Skill change only — `.claude/skills/ultraship/SKILL.md`. No `scripts/ultraship.py`
change, no `SCHEMA_VERSION` bump, no product code path, no hook change.
