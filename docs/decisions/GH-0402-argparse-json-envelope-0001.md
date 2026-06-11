# argparse usage errors emit the JSON error envelope (issue #402)

> Source: [#402](https://github.com/jswest/bartleby/issues/402)

A malformed flag/arg to a skill script used to fall through argparse's default
`ArgumentParser.error()` path: a raw `usage: ...` dump to stderr plus
`SystemExit(2)`. Agents parse skill output expecting *one* shape — the
`{"error", "code"}` JSON envelope on stdout — so a usage error was the lone
escape hatch that broke that contract.

Fix is a single seam, not a per-script change. `bartleby/skill_runner.py`
already runs every skill's `work()` inside a `try` that turns `SkillError` and
any stray exception into the envelope. The `parse_args(argv)` call sat *outside*
that `try`; it now moves *inside*, and a new `except SystemExit` arm catches
argparse's exit. The arm splits on exit code: a falsy code (`None`/`0` — what
argparse uses for `--help`/`-h`) is re-raised untouched so help still prints and
exits 0, while any non-zero code (the usage/argument errors) becomes
`{"error": "Invalid arguments. See --help for usage.", "code": "USAGE_ERROR"}`
on stdout with exit 1. Parsing happens before any DB/session is opened, so the
arm returns straight away with no conn/audit bookkeeping.

Deliberately **not** done: no `ArgumentParser.error()`/`.exit()` subclass spread
across the ~18 scripts — the runner is the one chokepoint every script funnels
through, so one arm covers them all (smallest fix that fits). argparse still
writes its own `usage:` line to stderr before raising; that's a separate stream
and harmless — the contract is about the parseable shape on stdout, which is now
always the envelope. The `USAGE_ERROR` code is an inline string constant, in
keeping with the runner's existing `INTERNAL_ERROR`/`NO_ACTIVE_PROJECT` codes
(no central code module exists). Two existing tests that asserted exit-2 on
argparse conflicts (`scan` `--count-by` vs `--preview/--brief`,
`list_documents` mutually-exclusive `--brief/--verbose` and `--sort` choices)
were updated to assert exit 1 + `USAGE_ERROR`. No schema change.
