Get a summary of a specific document.

Returns the cached LLM-generated summary if available. If no summary exists, returns the first ~15 chunks as an approximation. Check the `source` field in the response:
- `"summary"` — A proper LLM-generated summary
- `"first_chunks"` — Raw text from the beginning of the document (approximation)

**When to use:**
- To quickly understand what a document is about before diving in
- To decide which documents are relevant to the user's question
- After listing documents, to survey the most promising ones
