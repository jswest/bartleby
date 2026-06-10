# `bartleby scribe` exits non-zero when any ingest unit is unresolved

**Issue:** #311 (part of the v0.8.10 omnibus, #315)

## Context

`scribe`'s exit code is the only machine-readable signal a scripted caller
(`bartleby scribe ... && next-step`) has. Before this change, per-unit failures
only *warned*: even a run where every file failed fell through to
`console.complete("Done.")` and exited 0, so a run that ingested nothing read as
green and the chained step ran anyway.

## Decision

`main()` in `commands/scribe.py` now captures, before the connection closes:

```python
had_failures = incomplete_count > 0 or bool(failures)
```

and at the end of the run **suppresses the green "Done." and calls `sys.exit(1)`**
when `had_failures`. The fix stays in the thin entry point — the CLI dispatch
(`cli.py:_scribe`) is unchanged; `SystemExit` propagates through it.

The failure signal is the **union** of two distinct sources, because neither
alone is complete:

- `writer.failures()` — the `failed_ingests` ledger ("still-unresolved failed
  unit"). This is the *only* source that catches a **parse** failure, which
  leaves no document row and so never shows up in `incomplete_count`.
- `incomplete_count` — documents that parsed but still owe captions or a summary.

Both a will-retry failure and a capped one count as unresolved: the exit code
reflects "this run did not finish cleanly," not "this unit is permanently dead."

A no-op resume (nothing missing) and the early `return` paths (no supported
files) leave both terms zero, so they exit 0 — the success contract is
preserved.

## Why not narrow it to capped-only

A transient VLM-unavailable caption that *will* retry is still an unresolved unit
this run; a `&&`-chained caller should halt rather than proceed on a corpus that
isn't fully ingested. Re-running resumes the missing units and, once they land,
exits 0.

## No false positives from disabled features

`incomplete_count` does not fire on config-disabled stages: with vision off,
image files are skipped (`classify.py`) and embedded images aren't extracted
(`parsers.py`), so there are no null-analysis rows to count; the summary `owed`
term is already gated behind `summaries_enabled`. So a normal no-vision /
no-summary run still exits 0.
