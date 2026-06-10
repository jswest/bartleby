# Split `scribe.py` into `bartleby/ingest/` phase modules (issue #306)

> Source: [#306](https://github.com/jswest/bartleby/issues/306)

`bartleby/commands/scribe.py` had grown to ~1,600 lines owning everything the
ingest pipeline does: phase orchestration, per-type file parsing, EDGAR
unwrapping, image routing, provider/worker resolution, the progress tally, and
the caption/summarize passes. Two problems followed. First, every one of the
v0.8.10 omnibus's follow-on fixes (#307–#314) collided on this one file, so they
could not be worked in parallel. Second — the load-bearing one — the three
phases were *structurally asymmetric*: caption and summarize each carried the
`is_capped` retry-cap gate in a helper, while parse ran inline in `main()`
without it, which is exactly why the missing parse cap (#307) was invisible. This
is the Phase-0 enabling refactor for that omnibus: a behaviour-preserving split,
one PR, five green commits (the suite stayed green at each), no schema touch.

**The split.** Each ingest concern moves to its own module under
`bartleby/ingest/`, leaving `commands/scribe.py` a ~330-line entry point holding
only `main()` (the orchestration) and `_report_failures`:

- **`resolve.py`** — the provider resolvers (`_resolve_{llm,vision}_provider`) and
  worker-count resolvers (`_resolve_{max,caption,summarize}_workers`), plus
  `_required_hf_models`. Pure config→value functions, no DB.
- **`parsers.py`** — the DB-free parse stage: the per-type converters and
  `_parse_document` dispatcher, image routing (`_ImageRoute`,
  `_parse_image_routes`), the chunk/token plumbing (`_archive`,
  `_build_chunk_inputs`, `_token_count`), and the pool scaffolding (`ParseConfig`
  / `ParseRequest` / `ParseOutcome`, `_parse_request`, `_warm_worker`).
- **`classify.py`** — source bucketing: `_collect_files`, `_hash_file`,
  `_is_complete`, `_ResumeItem`, `_classify` (skip / resume / queue / in-run-dup).
- **`caption.py`** — phase 2 (`_analyze_image`, `_caption_from_analysis`,
  `_caption_all`).
- **`summary.py`** — phase 3 (`_summary_chunks`, `_summarize_all`). Kept separate
  from the existing `summarize.py` (which it calls) to avoid a writer↔summarize
  import cycle.
- **`parse.py`** — phase 1: `parse_all()` (the drain lifted out of `main()`) and
  the `DocUnit` it produces, which caption/summary carry forward.
- `_ProgressTally` joins the live-display machinery in **`progress.py`**.

**Behaviour-preserving means moves + import fixes only.** The single non-paste
edit was lifting `main()`'s phase-1 loop verbatim into `parse_all()`; everything
else is a paste with the call site re-qualified. Two deliberate import-style
changes were needed so the existing test monkeypatches keep biting after the
code moved module: the parse converters and `_summary_chunks` call
`embed.embed_texts(...)` **module-qualified** (so the `bartleby.ingest.embed`
patch covers them), and `main()` calls the resolvers / `parse_all` /
`caption._caption_all` / `summary._summarize_all` through their module objects.
`tests/test_scribe.py`'s patch strings were retargeted to the new namespaces
rather than papered over with re-export shims in `scribe.py` — a shim would be
exactly the dormant-old-path the repo's no-compat rule forbids.

**Layering.** The dependency edges form a DAG: `parse → {parsers, pool,
progress, writer}`, `caption → parse`, `summary → {parsers, summarize}`,
`classify → parsers`, and `scribe.main` → all of them. `parse.py` names
`classify._ResumeItem` only in an annotation, so that edge is a `TYPE_CHECKING`
import — no runtime `parse → classify` dependency.

The payoff lands next: with the three phases symmetric, #307's fix is a one-line
`is_capped('parse')` gate in `parse.py` mirroring `caption.py`/`summary.py`, and
each of #307–#314 now edits a distinct module.
