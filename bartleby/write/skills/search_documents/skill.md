---
name: search_documents
agents: [search_expert]
display:
  progress_message: "Searching documents..."
  completed_label: "Searched documents"
  context_arg: query
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
  exact_match:
    type: boolean
    description: "If true, search for the exact phrase using keyword matching only (no semantic search)"
    nullable: true
output_type: string
---

Search documents using hybrid retrieval (keyword + semantic matching).

Combines full-text keyword search and meaning-based vector search, then merges and re-ranks results for best coverage. You don't need to choose between search modes — this tool handles both.

**Best for:** Any search query — exact terms, natural language questions, conceptual queries, or a mix. Prefer hybrid search (the default) unless you have a specific reason to use exact matching.

**Exact match mode:** Set `exact_match: true` to search for the query as an exact phrase using keyword matching only. This disables semantic search and re-ranking. Use this when:
- You need to find a specific term, name, or phrase (e.g., "Clean Air Act Section 109")
- Counting how many times something is mentioned
- The hybrid results are too broad or off-topic

**Important:** Searches within individual text chunks (~800 characters each). For multi-term queries, keep it focused:
- Use 1-3 specific keywords or a short natural language question
- Results are ranked by combined relevance across both search methods

**Result fields:** Each result includes `ref` (citation number), `chunk_id`, `body` (preview), `score`, `page_number`, `document_id`, `section_heading`, and `content_type` when available. If `body_truncated` is true, use `get_chunk_window` to read the full passage.

**Examples:**
- Keywords: `carbon emissions`
- Question: `what are the main findings?`
- Specific: `Table 3 methodology`
- Conceptual: `environmental impact on communities`
- Exact: `exact_match=true`, query=`"Clean Air Act"`
