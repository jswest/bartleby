# An HTML page saved as `.pdf` is rejected with a clear reason, never rerouted into the HTML pipeline

Issue #235. Upstream scrapers (the NY DPS DMM portal, via the `puc` repo) saved
HTTP-error / `text/html` responses to `*.pdf` files. Bartleby trusted the `.pdf`
extension, handed the bytes to the pdfplumber backend, and died deep in pdfminer
with the cryptic `No /Root object! - Is this really a PDF?` — which sends you
hunting for PDF corruption instead of a bad download. ~0.3% of one 9k-PDF corpus
was contaminated this way.

This is the mirror image of #78, which handled *extension missing/wrong but
content is a valid supported type* by sniffing and **recovering** it. Here the
extension is present and claims `.pdf` but the content is a **non-document** (an
HTML error page). Two responses were on the table:

- **(a)** Sniff the content to `.html` and route it into the Docling HTML
  pipeline — the symmetric "recover" move.
- **(b)** Detect the mismatch at PDF dispatch and **reject** with an actionable
  failure reason; do not reroute.

## Decision

Go with **(b)** — detect and reject.

Recovering (a) would *succeed*: it would ingest a portal error page
("Exception occured while servicing the request…") as a first-class document,
polluting the corpus and its embeddings. Garbage-in is worse than a clean skip.
So the guard (`reject_if_html` in `bartleby/ingest/pdfplumber.py`, called once at
the PDF-dispatch point in `scribe._parse_document`) raises a typed `NotAPdfError`
with `not a PDF — file contains an HTML page (likely a failed download or portal
error page)`. That rides the existing parse-failure path into `failed_ingests`
and prints in place of the pdfminer string. No reroute, no schema change.

The guard is deliberately **conservative**: it returns immediately on a `%PDF`
head and only rejects content that *clearly* begins with an HTML marker
(`<!doctype html`, `<html`). A head that is merely not-`%PDF` is still handed to
the backend, which tolerates real-but-slightly-malformed PDFs.

## The open question, resolved: failure, not quiet skip

The issue asked whether a detected HTML-error-page should count as a *failure*
(surfaced loudly, against `failed_ingests`) or a *quiet skip*. We chose
**failure-with-clear-reason**. A silent skip is exactly the "partial corpus, no
signal" trap #78 called out — you want the loud signal so you go fix the scrape
upstream. The real root cause (the scraper writing error pages to `.pdf`) lives
in the `puc` repo, which has no GitHub remote here; this decision only stops
Bartleby from misdiagnosing the symptom.
