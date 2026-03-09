---
name: get_chunk_window
agents: [search_expert, research]
display:
  progress_message: "Reading passage..."
  completed_label: "Read passage"
inputs:
  chunk_id:
    type: string
    description: "Chunk ID from a search result"
  window_radius:
    type: integer
    description: "Chunks before/after the anchor (default 3)"
    nullable: true
output_type: string
---

Read a window of text around a specific search hit.

Use this when you find a relevant chunk via search and need more surrounding context. Provide the `chunk_id` from any search result to grab nearby text (default ~3 chunks on either side, roughly 2400 characters total).

**When to use:**
- A search result looks relevant but is truncated
- You need to understand the context around a key finding
- You want to read a continuous passage without fetching the entire document
