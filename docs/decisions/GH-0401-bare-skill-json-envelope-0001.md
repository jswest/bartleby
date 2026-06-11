# Bare `bartleby skill` returns a JSON error envelope (issue #401)

> Source: [#401](https://github.com/jswest/bartleby/issues/401)

The dispatcher in `bartleby/commands/skill.py` lumped the empty-argv case in with
`-h/--help`: `if not argv or argv[0] in ("-h", "--help")` both printed usage help
to stdout and returned exit 0. So `bartleby skill` with **no script name** broke the
skill contract that every other error path honors — a `{"error", "code"}` JSON
envelope on stdout, exit non-zero — and instead emitted prose on stdout with a
success code, which an agent parsing stdout as JSON would choke on.

Fix splits the two cases in `dispatch`. `-h/--help` keeps its old behavior verbatim
(usage help to stdout, return → exit 0). Empty argv now mirrors the existing
unknown-skill path: usage prose goes to **stderr only** via `_print_help(sys.stderr)`,
and a `{"error": "No skill script given. Available: …", "code": "MISSING_SKILL"}`
envelope is written to stdout (compact JSON, matching the `UNKNOWN_SKILL` path's
`separators` and trailing newline) followed by `sys.exit(1)`. The new
`MISSING_SKILL` code names the missing-script condition distinctly from
`UNKNOWN_SKILL` (a *named* script that isn't registered).

Smallest fix that fits — no new helper, no shared envelope abstraction; the two
error sites are short and intentionally parallel. Covered by
`tests/test_skill_dispatch.py`: bare argv → `MISSING_SKILL` envelope on stdout + exit
1 + prose on stderr only; `-h`/`--help` → exit 0 with usage on stdout; unknown name →
`UNKNOWN_SKILL` envelope unchanged. No schema change.
