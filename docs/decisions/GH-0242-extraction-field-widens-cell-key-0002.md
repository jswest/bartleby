# The `extraction` field is a first-class cell-key dimension in the benchmark.

The benchmark cell was previously `(provider, model, doc)`. With Docling as an
opt-in extraction backend, a Docling-extracted source for the same doc produces
meaningfully different input text — different chunking granularity, Markdown
export instead of raw text, section-heading contextualization. Treating it as a
separate doc-id would distort the equal-per-doc weighting in quality aggregation;
making it a cell-key dimension keeps the doc axis clean while still separating the
runs in the store.

**What changed.** The cell key is now `(provider, model, doc, extraction)`:

- Run records gain an `extraction` field (default `"pdfplumber"`).
- Judgment records gain an `extraction` field (copied from the run's extraction).
- Result filenames: `<provider>_<model>_<doc>.jsonl` for pdfplumber (unchanged,
  backward-compatible); `<provider>_<model>_<doc>_x-<extraction>.jsonl` for named
  variants. The `_x-` prefix avoids ambiguity with doc-ids that already contain
  hyphens.
- Judgment filenames follow the same pattern.
- `BenchmarkRoot.source_path(doc_id, extraction)` returns the source cache path:
  `sources/<doc-id>.txt` for pdfplumber, `sources/<doc-id>-<extraction>.txt` for
  named variants.
- `bartleby benchmark summarize --extraction <name>` runs against the named
  extraction's source (which must exist as a pre-committed fixture or prior run).
- `bartleby benchmark leaderboard --extractions a,b` filters to those extraction
  variants; default is all on record.

**Heterogeneity invariant preserved.** A cell's records must never mix `source_sha`
regimes. Because the cell key now includes `extraction`, and each extraction variant
has its own source file (and therefore its own sha), this invariant is
automatically satisfied — a pdfplumber run and a docling run for the same model+doc
are in different cells, never averaged together.

**Docling variant is fixtured.** Running Docling live in CI requires heavy model
downloads (layout/table detection models). The committed fixture
`benchmarks/sources/psc-order-0109-docling.txt` was produced once by running the
production `bartleby.ingest.docling.convert` on the corpus PDF; it is tracked in
git so subsequent benchmark invocations (`--extraction docling`) read it without
invoking Docling. If Docling's extraction quality changes, regenerate the fixture
and let `source_sha` detect the drift.

**Aggregation.** The leaderboard's quality table aggregates across extractions for
the same `(ref, doc)` pair by averaging scored cells. This is appropriate when
comparing overall quality per doc across extraction variants. The overall quality
per model averages across all `(doc, extraction)` cells equally.
