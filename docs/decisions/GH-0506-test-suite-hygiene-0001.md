# Vacuous test assertions are made falsifiable and the embed argv test stops loading the real model (issue #506)

> Source: [#506](https://github.com/jswest/bartleby/issues/506)

Several assertions across the suite passed regardless of the behavior they
claimed to pin, so a real regression would have sailed through them. This bundle
makes each one able to actually fail, removes the dead imports/params that had
accumulated around them, closes an `os.environ` leak in `test_quiet.py`, and
takes the real BAAI embedding-model load out of `test_embed.py`'s default path.
Purely test-quality work — no production code changed.

**Why these were vacuous, and the fix in each case:**

- **Overlap (`test_ingest_text.py`).** The input was `"abcdefghij" * 100` — a
  10-char period. `prev[-50:] in curr` then matched on the *pattern's*
  periodicity, not on bytes the chunker actually copied across the overlap; it
  held even at zero overlap. Replaced with a monotonic, whitespace-free digit
  stream (`"".join(f"{i:04d}" for i in range(500))`) where every 50-char window
  is unique and appears exactly once, so a found tail proves a real overlap. No
  spaces means the chunker's per-piece `.strip()` can't shift the boundary out
  from under the slice. Verified the assertion now fails when overlap is forced
  to 0.
- **RRF order (`test_skill_search.py`).** `assert ids_in_order[0] in (10, 30)`
  admitted either of the top two ids, so a tie-reversal regression passed.
  Pinned the full deterministic order `[10, 30, 20, 50, 40]` from the known
  two-list input.
- **Lane callbacks (`test_scribe.py`).** The summarize-lane assertion was
  `phase.lanes == [(phase.lanes[0][0], "doc.txt", "summarizing")]` — the lane
  key was read back out of the result and compared to itself, so a regression
  that dropped or mangled the key passed. With `workers=1` the pass runs inline
  on the test thread, so the key is a *known* value: both the summarize and the
  caption lane tests now assert the exact tuple keyed on
  `threading.get_ident()`.
- **`os.environ` leak (`test_quiet.py`).** `setup_quiet_third_party` writes the
  offline flags and the noise-control vars straight into `os.environ`
  (`setdefault` / `os.environ[...] =`), which monkeypatch does not track. A test
  that flipped `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` to `"1"` left them set
  for every later test, so `offline_blocked` could read a stale `"1"` it never
  set. An autouse module fixture now snapshots and restores `os.environ` around
  every test in the file, keeping each case hermetic and the offline assertions
  falsifiable rather than passing on a leaked value.
- **Real-model load (`test_embed.py`).** `test_embed_command_subprocess_listform`
  shelled out to `uv run bartleby embed ...`, which loaded the real BAAI model
  in a subprocess and dominated suite runtime. The property it cared about —
  list-form argv hands the query to the embedder verbatim, with shell
  metacharacters inert — is now asserted in-process by monkeypatching
  `embed_texts` to record exactly what `main` forwards, then checking the
  adversarial `"hello world; rm -rf /"` arrives un-split. The
  no-shell-interpretation guarantee of list-form `subprocess.run` is Python's,
  not ours to re-test on every suite run. Chosen over a `slow` marker because no
  such marker convention exists in `pyproject.toml`; inventing one was out of
  scope.

Dead weight removed alongside: four unused `from bartleby.commands import scribe
as scribe_module` locals in `test_scribe.py`, an unused `ChunkInput,
insert_image_chunks` import in `test_skill_search.py`, and unused `monkeypatch`
params in `test_skill_save_finding.py` and `test_quiet.py`.

Tests: full suite green (1037 passed), and the run no longer loads the real
embedding model — wall time dropped from ~2min (dominated by the subprocess
test) to ~18s. No schema change.
