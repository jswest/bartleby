#!/usr/bin/env bash
# PreToolUse(Bash) guard for the /ship workflow's "never commit to main" rule.
# A skill can only *advise*; this hook *enforces*: it refuses `git commit` and
# `git push` whenever the branch in the directory the command runs in is the
# default branch. Everything else is allowed — including `git pull --ff-only` on
# main, which the cleanup step needs.
#
# Effective directory: the hook starts from the event's cwd, then replays a
# leading `cd <dir>` within the command and honors a per-invocation `git -C
# <dir>`. This matters because the agent drives git as `cd <worktree> && git
# commit` without changing the session cwd — checking the raw event cwd would
# wrongly see `main` every time.
#
# Matching: the command is split on shell separators (; & | newline); a segment
# is gated only if it starts with `git` AND its *subcommand* (the first
# non-option token, skipping git's global options) is commit or push. So
# `git log --grep commit` and `echo 'git commit'` are NOT blocked.
#
# Push is judged by *where it writes*, not the current branch. A `git push`
# carrying an explicit positional refspec is gated on that refspec's destination
# ref — blocked only when it resolves to main/master (`origin main`, `HEAD:main`,
# `develop:main`, `+main`, `refs/heads/main`, `--delete main` all count). So an
# explicit non-main push (`push -u origin issue/42-foo`) is allowed even from
# main — which the ship workflow needs to create an omnibus branch. A push with
# no positional refspec (bare `git push`, `git push origin`, `--all`, `--mirror`)
# would resolve through the configured upstream / push.default; we don't parse
# that, so it falls back to the current-branch check, exactly like commit.
#
# Known limitation: a separator (`;` `&` `|`) or a `cd`/git line *inside quotes
# or a here-doc* is still treated structurally (no real shell parsing), so it can
# mis-resolve. Rare in practice and overridable by running the command yourself
# with `! <cmd>`. This is a backstop, not a sandbox. Targets macOS bash 3.2.
set -euo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
event_cwd=$(printf '%s' "$input" | jq -r '.cwd // ""')
[ -n "$event_cwd" ] || event_cwd=$(pwd)

# Tokenize a segment on whitespace regardless of the caller's IFS (the segment
# loop sets IFS=$'\n'; without a local reset `read -a` would dump the whole
# segment into toks[0] and silently break detection).
_tokenize() { local IFS=$' \t\n'; read -r -a TOKS <<<"$1"; }

# First non-option token after `git` (the subcommand), or empty.
git_subcommand() {
  local -a TOKS; _tokenize "$1"
  [ "${TOKS[0]:-}" = "git" ] || return 0
  local i=1
  while [ "$i" -lt "${#TOKS[@]}" ]; do
    case "${TOKS[$i]}" in
      -C|-c|--git-dir|--work-tree|--namespace|--exec-path) i=$((i + 2)) ;;
      -*) i=$((i + 1)) ;;
      *) printf '%s' "${TOKS[$i]}"; return 0 ;;
    esac
  done
}

# Value of `git -C <path>` for this segment, or empty. Only called after
# git_subcommand matched commit/push, so TOKS[0] is already known to be `git`.
git_dir_opt() {
  local -a TOKS; _tokenize "$1"
  local i=1
  while [ "$i" -lt "${#TOKS[@]}" ]; do
    case "${TOKS[$i]}" in
      -C) printf '%s' "${TOKS[$((i + 1))]:-}"; return 0 ;;
      -c|--git-dir|--work-tree|--namespace|--exec-path) i=$((i + 2)) ;;
      -*) i=$((i + 1)) ;;
      *) return 0 ;;
    esac
  done
}

# Destination ref of a `git push` segment's positional refspec, normalized, or
# empty when the push carries no positional refspec (bare push / remote-only /
# --all / --mirror — those fall back to the branch check, preserving prior
# behavior). Only called after git_subcommand matched push, so TOKS[0] is git.
push_dest_ref() {
  local -a TOKS; _tokenize "$1"
  local i=1 n=${#TOKS[@]}
  # Skip git's global options to reach the `push` subcommand token.
  while [ "$i" -lt "$n" ]; do
    case "${TOKS[$i]}" in
      -C|-c|--git-dir|--work-tree|--namespace|--exec-path) i=$((i + 2)) ;;
      -*) i=$((i + 1)) ;;
      *) break ;;
    esac
  done
  i=$((i + 1))   # step past `push`
  # Walk push args, collecting positionals: #1 is the remote, #2 the refspec.
  # Skip the push options that consume the following token, lest their value be
  # mistaken for the remote (`push -o ci.skip origin main`).
  local pos=0 refspec=""
  while [ "$i" -lt "$n" ]; do
    case "${TOKS[$i]}" in
      -o|--push-option|--repo|--receive-pack|--exec) i=$((i + 2)) ;;
      -*) i=$((i + 1)) ;;
      *)
        pos=$((pos + 1))
        [ "$pos" -eq 2 ] && { refspec="${TOKS[$i]}"; break; }
        i=$((i + 1))
        ;;
    esac
  done
  [ -n "$refspec" ] || return 0
  # Normalize to the destination ref: drop a leading '+' (force), keep the part
  # after the last ':' (src:dst → dst), drop a refs/heads/ prefix.
  refspec="${refspec#+}"
  refspec="${refspec##*:}"
  refspec="${refspec#refs/heads/}"
  printf '%s' "$refspec"
}

# Absolute path of $2 resolved from base $1, or empty if it doesn't exist.
resolve_dir() { (cd "$1" 2>/dev/null && cd "$2" 2>/dev/null && pwd) || true; }

eff_cwd="$event_cwd"
blocked=0
old_ifs=$IFS
IFS=$'\n'
for seg in $(printf '%s' "$cmd" | tr ';&|' '\n'); do
  seg="${seg#"${seg%%[![:space:]]*}"}"   # strip leading whitespace
  declare -a TOKS; _tokenize "$seg"

  # A leading `cd <dir>` moves the effective directory for later segments.
  if [ "${TOKS[0]:-}" = "cd" ] && [ -n "${TOKS[1]:-}" ]; then
    moved=$(resolve_dir "$eff_cwd" "${TOKS[1]}")
    [ -n "$moved" ] && eff_cwd="$moved"
    continue
  fi

  sub=$(git_subcommand "$seg")
  case "$sub" in
    commit|push) ;;
    *) continue ;;
  esac

  # An explicit push refspec is judged by its destination ref, not the branch:
  # block only when it writes main/master. A refspec-less push falls through to
  # the branch check below, like commit.
  if [ "$sub" = "push" ]; then
    dest=$(push_dest_ref "$seg")
    if [ -n "$dest" ]; then
      case "$dest" in main|master) blocked=1 ;; esac
      continue
    fi
  fi

  check_dir="$eff_cwd"
  dopt=$(git_dir_opt "$seg")
  if [ -n "$dopt" ]; then
    r=$(resolve_dir "$eff_cwd" "$dopt"); [ -n "$r" ] && check_dir="$r"
  fi

  branch=$(git -C "$check_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
  case "$branch" in main|master) blocked=1 ;; esac
done
IFS=$old_ifs

if [ "$blocked" -eq 1 ]; then
  echo "BLOCKED: refusing 'git commit'/'git push' on the default branch. The ship workflow does all work on a sibling worktree branch (issue/<N>-<slug>); create or switch to a feature branch first. (Override: run the git command yourself with '! <cmd>'.)" >&2
  exit 2
fi
exit 0
