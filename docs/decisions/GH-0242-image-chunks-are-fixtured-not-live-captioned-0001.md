# Image-bearing benchmark docs use pre-committed fixtures, not live VLM captioning.

The benchmark's image-bearing corpus document (`neada-heating-0461`) exercises
the full interleaving path (`writer.summary_input`'s `[Image on page N]\n<text>`
interleave) without requiring the local VLM caption pipeline to run as a
benchmark precondition.

**Why fixtures, not live captioning.** The benchmark's value is comparing model
*summarization* quality, not caption quality. Requiring Ollama's vision model at
benchmark time would (a) make the benchmark non-reproducible across machines that
don't have a vision backend configured, (b) make caption quality a confound in
summarization scores, and (c) couple benchmark results to the VLM's version.
Pre-committing the source text lets every machine reproduce the exact same
summary input.

**What is fixtured.** The source text under
`benchmarks/sources/neada-heating-0461-image-fixture.txt` is produced by
running the pdfplumber extraction pipeline on the PDF, chunking all text pages
exactly as production does (same `chunk_text` + page-order interleaving), and
then substituting hand-written descriptions for the five embedded data-table
images. The descriptions were written from visual inspection of each image crop
(via `pdfplumber`'s `EmbeddedImage.png_bytes`). The format exactly mirrors
`writer.summary_input()`: `[Image on page N]\n<caption text>` in page-sorted
order between the surrounding text chunks.

**Provenance.** `source_sha` pins every run and judgment record to the
committed fixture text. If the fixture is ever updated (e.g. to correct a
caption), `source_sha` changes, leaderboard heterogeneity warnings fire for
any cell that mixes old and new runs, and old runs cannot be judged against
new text (the judge refuses with a sha mismatch error). This is exactly the
same drift-detection mechanism the existing pdfplumber-extracted sources use.

**Accepted tradeoff.** Fixtured captions slowly drift from the current
production VLM's output quality as models evolve. That is acceptable — the
benchmark is measuring summarizer performance on a fixed input, not tracking
caption drift. Anyone wanting to update the captions can regenerate the
fixture and bump the `source_sha`.

**How to use.** Run `bartleby benchmark summarize --extraction image-fixture`
to use the fixtured source for `neada-heating-0461`. The other corpus docs run
with the default `pdfplumber` extraction, so the two extraction values coexist
in the results store and can be filtered independently with `--extractions` on
the leaderboard.
