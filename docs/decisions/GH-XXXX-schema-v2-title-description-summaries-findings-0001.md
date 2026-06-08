# Schema v2 — title/description on summaries and findings

`summaries.{title, description}` and `findings.{title, description}` (all NOT NULL) so `list_documents` and finding browsing aren't filename-only. Summarizer returns all three fields in one structured-output call — we don't pay for the document text three times.
