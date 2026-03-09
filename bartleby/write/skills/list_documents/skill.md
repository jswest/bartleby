---
name: list_documents
agents: [search_expert]
display:
  progress_message: "Listing documents..."
  completed_label: "Listed documents"
inputs: {}
output_type: string
---

List all documents in the database with metadata.

Returns document IDs, filenames, page counts, chunk counts, titles (if summarized), and whether a summary exists.

**When to use:**
- At the start of research to understand what's in the corpus
- To find document IDs for targeted searches or reading
- To check which documents have summaries available
