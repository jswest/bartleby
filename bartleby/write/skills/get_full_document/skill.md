---
name: get_full_document
agents: [search_expert]
inputs:
  document_id:
    type: string
    description: "Document ID from search results"
  start_chunk:
    type: integer
    description: "Zero-based index to start from (default 0)"
    nullable: true
  max_chunks:
    type: integer
    description: "Max chunks to return (capped at 100)"
    nullable: true
output_type: string
---

Retrieve chunks from a specific document with pagination.

Use this to read through a document sequentially. Provide the `document_id` from a search result or document listing. Use `start_chunk` to page through long documents.

**When to use:**
- You need to read a document from the beginning
- You want to continue reading after a previous call
- You need comprehensive coverage of a specific document

**Pagination:** Returns up to 100 chunks per call. Check `has_more` and `next_start_chunk` in the response to continue reading.
