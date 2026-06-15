#!/usr/bin/env bash
# SessionStart hook — point BARTLEBY_HOME at a per-worktree sandbox so a coding
# agent working in an isolated git worktree never reads or writes the
# developer's live ~/.bartleby corpora. See GH-0393.
#
# Mechanism vs. policy: the bartleby *product* only honors BARTLEBY_HOME (a
# generic state-dir override, like PGDATA/GIT_DIR). This hook is the *tooling*
# that sets it — so the product never has to learn what a git worktree is. The
# user's primary checkout is deliberately left alone: it resolves to the live
# ~/.bartleby exactly as before, so this is scoped to agent (worktree) sessions.
#
# Detection: a linked git worktree has `.git` as a FILE (containing `gitdir:`);
# the primary checkout has `.git` as a DIRECTORY. That one distinction is "am I
# an isolated agent worktree?" — no extra git plumbing, no knowledge of how the
# worktree was made (sibling `../bartleby-issue-*`, `.claude/worktrees/<id>` —
# all are linked worktrees).
#
# Caveat: this only takes effect where the harness runs SessionStart hooks and
# provides $CLAUDE_ENV_FILE. Where it does not fire the behavior is unchanged
# (live ~/.bartleby) — it degrades safe, it just stops being automatic. Never
# crash the session: every failure path exits 0.

payload="$(cat 2>/dev/null)"
dir="$(printf '%s' "$payload" | sed -n 's/.*"cwd"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
[ -z "$dir" ] && dir="$PWD"

root="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)"
[ -z "$root" ] && exit 0            # not in a git repo — nothing to isolate

# Primary checkout (`.git` is a dir) → leave BARTLEBY_HOME unset → live data.
[ -f "$root/.git" ] || exit 0

# Linked worktree → a stable per-worktree sandbox, reused across the session's
# many bartleby invocations (a project created in one call is visible in the
# next) and trivially deletable wholesale (rm -rf ~/.bartleby-worktrees).
hash="$(printf '%s' "$root" | shasum 2>/dev/null | cut -c1-12)"
[ -z "$hash" ] && hash="$(printf '%s' "$root" | cksum | tr -dc '0-9')"
sandbox="$HOME/.bartleby-worktrees/$hash"

[ -n "${CLAUDE_ENV_FILE:-}" ] && \
  printf 'export BARTLEBY_HOME=%s\n' "$sandbox" >> "$CLAUDE_ENV_FILE"
exit 0
