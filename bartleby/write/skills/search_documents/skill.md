---
name: search_documents
agents: [search_expert]
inputs:
  query:
    type: string
    description: "Search query (keywords or natural language)"
  limit:
    type: integer
    description: "Maximum results to return (default 3, max 5)"
    nullable: true
  document_id:
    type: string
    description: "Optional document ID to search within"
    nullable: true
output_type: string
---

Search documents using hybrid retrieval (keyword + semantic matching).

Combines full-text keyword search and meaning-based vector search, then merges and re-ranks results for best coverage. You don't need to choose between search modes — this tool handles both.

**Best for:** Any search query — exact terms, natural language questions, conceptual queries, or a mix.

**Important:** Searches within individual text chunks (~800 characters each). For multi-term queries, keep it focused:
- Use 1-3 specific keywords or a short natural language question
- Results are ranked by combined relevance across both search methods

**Result fields:** Each result includes `ref` (citation number), `chunk_id`, `body` (preview), `score`, `page_number`, `document_id`, `section_heading`, and `content_type` when available. If `body_truncated` is true, use `get_chunk_window` to read the full passage.

**Examples:**
- Keywords: `carbon emissions`
- Question: `what are the main findings?`
- Specific: `Table 3 methodology`
- Conceptual: `environmental impact on communities`
