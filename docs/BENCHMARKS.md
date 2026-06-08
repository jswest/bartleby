# Ingest benchmarks

A living record of `bartleby scribe` ingest performance: how to measure it
reproducibly, and dated results to compare against. The concurrency work tracked
in [#169](https://github.com/jswest/bartleby/issues/169) needs a durable
baseline ("pages/sec at N workers"), and the per-stage split tells us *which*
stage is worth parallelizing. Append new runs under [Results](#results) — don't
overwrite; the point is comparison over time.

## How to run

`bartleby scribe --timings` (added in
[#162](https://github.com/jswest/bartleby/issues/162)) times each document's
`prep` / `parse` / `embed` / `caption` / `summarize` stages, prints the per-doc
split to stderr, and emits an aggregate as one JSON object to stdout.

```sh
# Fresh project — ingest dedups, so a re-run on a used project times nothing.
bartleby project delete bench -y && bartleby project create bench
bartleby scribe --project bench --files <sample> --timings > bench.json
```

The aggregate JSON carries `docs`, `pages`, `wall_clock_s`, `docs_per_s`,
`pages_per_s`, and a per-stage breakdown (`total_s`, `pct`, `mean_s`). Rates use
whole-run wall-clock, so inter-doc overhead counts.

**Always record the resolved config with the result.** Stage costs are entirely
model- and hardware-specific — converters, providers, models, and the vision
dimension knobs all move the numbers.

### Gotchas that silently corrupt a run

- **The summary model must emit structured JSON.** A model that returns prose
  fails `DocumentSummary` schema validation, which *raises out of the document*
  before its timing is recorded — so the doc isn't counted and you get
  `docs: 0`. (`qwen3.6:35b-mlx` failed 10/10 this way; `qwen3:30b` worked.)
- **Tesseract needs a usable `TMPDIR`.** `pytesseract` writes a temp file under
  `$TMPDIR` and shells out to `tesseract`. If that path isn't usable by the
  subprocess, the OCR pre-pass throws `UnicodeDecodeError` and *aborts the whole
  image* — the VLM never runs, and captioning silently produces nothing while
  looking near-instant. If your caption stage reads as ~free, check this before
  believing it. (Seen under a harness that set `TMPDIR=/tmp/claude-501`.)

## Results

### 2026-06-07 — first baseline (10-doc NTSB sample)

**Sample:** 10 docs / 42 pages from `ntsb-rr-chemicals`, with 18 images
captioned in this run (~1.8 img/doc — image-light). (The image *audit* below
counts 23 images for the same 10 docs: it was run against the persistent
`ntsb-rr-chemicals` copies of those docs, since the benchmark project was
ephemeral, and that earlier ingest used a different converter that extracted
slightly more images. The 18 vs 23 gap is converter extraction, not a counting
error.)
**Config:** `pdf_converter=docling`; summary `qwen3:30b`; vision `qwen3-vl:30b`;
all local Ollama, serialized; `vision_min_dimension=32`, `vision_max_dimension`
1024 (images stored at 512px long-edge from an earlier ingest).
**Throughput:** 0.020 docs/sec · 0.082 pages/sec (512s wall).

| Stage | total | % of stage time | mean/doc |
| --- | --: | --: | --: |
| parse (docling) | 58.6s | **11.5%** | 5.9s |
| embed (BGE) | 3.6s | 0.7% | 0.4s |
| caption (VLM) | 148.2s | **28.9%** | 14.8s (~8.2s/image) |
| summarize (LLM) | 301.6s | **58.9%** | 30.2s |
| prep (hash+archive) | 0.007s | ~0% | — |

**The #162 baseline question — "docling parse vs. ~55 captions/doc?" — answered:
neither.** On this sample **summarization dominates (59%)**, captioning is second
(29%), and parse is only **12%**. The two LLM/network stages are ~88% of the
work.

Per-doc shape:

- Text-heavy docs → summarize is essentially the whole cost (a 1-page doc:
  0.7s parse + 8.2s summarize).
- Largest doc (13pp): 291.6s ≈ 40s parse + 80s caption + **170s summarize**
  (plus ~1s embed) — summarize scales hard with document length.
- Caption scales per-image at ~8.2s; this sample is image-light. At the issue's
  ~55-img/doc worst case that's ~450s of captioning on one doc — caption is the
  **tail risk** even though it's second here.

**Implication for [#169](https://github.com/jswest/bartleby/issues/169):** a
parse pool ([#165](https://github.com/jswest/bartleby/issues/165)) alone caps at
~12% upside. The real wins are decoupling/batching the LLM stages —
[#167](https://github.com/jswest/bartleby/issues/167) (summarize),
[#166](https://github.com/jswest/bartleby/issues/166) (caption),
[#168](https://github.com/jswest/bartleby/issues/168) (Batch).

#### Image audit (23 images, on the persistent `ntsb-rr-chemicals` copies)

All 23 were `kind=scene` (VLM-described); **none produced Tesseract OCR text**.

- **~6 are real, valuable figures** — a corroded-weld close-up with a scale
  ruler, engineering diagrams, a flood-stage chart. Descriptions are substantive
  (600–900 chars) and accurate.
- **~17 are thin strips from one scanned table** (a material-test report sliced
  into rows; `min_edge` 36–78). Their VLM "descriptions" are near-useless — e.g.
  *"vertically oriented text in a stylized handwritten-style font, rotated ~90°"*
  — the model can't read the sliver and paraphrases vaguely. This is the
  "55 captions/doc" pathology in miniature.

Threshold effects on this sample (acted on in
[#179](https://github.com/jswest/bartleby/issues/179)):

- **`vision_min_dimension` 32 → 64** drops 10/23 images, all low-value strips
  (saves ~80s on the one table doc, loses ~nothing). Raising to ~80 would drop
  all 17 strips (74%) and keep every real figure.
- **`vision_max_dimension` 1024 → 512** is supported by direct evidence: this
  corpus was captioned *at* 512px and the figure descriptions are fully
  substantive. ~Quarters the VLM pixel count with no visible quality loss for
  scene description. (Detail-dense OCR targets want higher res — but those belong
  on the OCR/table path, not VLM description.)

*Caveat: one image-light corpus with the junk concentrated in a single
scanned-table doc — enough to show direction, not to set blessed defaults. A
broader audit could push these further.*
