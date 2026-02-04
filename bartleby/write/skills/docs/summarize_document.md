Generate and cache an LLM summary for a document.

Use this only when `get_document_summary` returns a first-chunk approximation (`source: "first_chunks"`) and you need a better understanding of that document. The summary is cached for future use.

**When to use:**
- `get_document_summary` returned an approximation, not a real summary
- The document appears important to the user's question
- You need a clear overview before searching within the document

**Note:** This makes an LLM call and may take a few seconds.
