Retrieve chunks from a specific document with pagination.

Use this to read through a document sequentially. Provide the `document_id` from a search result or document listing. Use `start_chunk` to page through long documents.

**When to use:**
- You need to read a document from the beginning
- You want to continue reading after a previous call
- You need comprehensive coverage of a specific document

**Pagination:** Returns up to 100 chunks per call. Check `has_more` and `next_start_chunk` in the response to continue reading.
