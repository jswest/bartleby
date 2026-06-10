# A sparse-page OCR failure degrades to the VLM, never fails the parse (issue #309)

> Source: [#309](https://github.com/jswest/bartleby/issues/309)

`pdfplumber.convert` runs Tesseract as the cheap first pass for sparse PDF pages:
if the OCR clears the length + confidence bar the page is treated as text,
otherwise the page render routes to the VLM. The classifier call
(`ocr_module.run`) was **unguarded**, so a Tesseract failure — a broken install,
a locked-down `TMPDIR`, the exact #43 scenario — propagated out of `convert` and
recorded the **entire document** as a parse failure, discarding every
extractable non-sparse page.

This is asymmetric with `images.analyze`, which made the opposite, deliberate
call for the identical situation: OCR there is only a classifier, so a Tesseract
crash degrades to "couldn't classify → route to the VLM" rather than dropping the
image. The decision here is to make both call sites obey the **same invariant**:

> **OCR is only a classifier. A broken Tesseract degrades that call site to the
> VLM route — it never fails ingest.** A busted install just means everything
> routes to the VLM (slower, but complete).

So `convert` now wraps the classifier call in `try/except`, sets `ocr_result =
None` on any exception (the existing `else` branch already routes a `None` result
through the VLM), and surfaces one warning. Non-sparse pages are untouched and
chunk normally; the document parse succeeds.

**Two supporting changes.**

- **`ocr.run` also catches `pytesseract.TesseractNotFoundError`.** It already
  re-wrapped `TesseractError` / `UnicodeDecodeError` into a legible `RuntimeError`
  pointing at the usual `TMPDIR` culprit (#43), but the *most common* breakage —
  the `tesseract` binary simply missing — raises `TesseractNotFoundError`, which
  skipped the re-wrap. It now gets the same legible cause.

- **The warn-once mechanism is consolidated into `console.warn_once(key,
  message)`.** Both `images.py` and `pdfplumber.py` previously carried a
  byte-identical `_ocr_degraded_warned` global + `_warn_ocr_degraded_once` helper
  differing only in message string. Two independent flags meant a mixed run
  (captioning images *and* parsing PDFs) with a broken Tesseract warned **twice**,
  defeating the "warn once" intent both claimed. The shared helper, keyed by a
  caller-supplied string, lets both surfaces pass the same `"ocr_degraded"` key so
  the degradation is surfaced once *per process across surfaces*, each keeping its
  own tailored message.

Reinforces [tesseract-owns-image-transcription / VLM-owns-description](GH-XXXX-tesseract-owns-image-transcription-vlm-owns-description-0001.md):
Tesseract's role is the cheap shortcut, never a hard dependency of ingest.
