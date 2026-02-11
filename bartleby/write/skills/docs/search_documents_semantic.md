Search documents using semantic similarity (meaning-based vector search).

**Best for:** Conceptual queries, finding related content, paraphrases, questions.

Unlike keyword search, this finds chunks with similar *meaning* even if they don't contain the exact words. Good for natural language questions and multi-concept queries.

**Result fields:** Results include `section_heading` (heading hierarchy, e.g. "Introduction > Background") and `content_type` (text/table/code/formula/list/picture) when available. These are populated when documents are processed with `--docling`.

**Examples:**
- `what are the main findings?`
- `how does the methodology work?`
- `arguments for and against the proposal`
- `environmental impact on communities`
