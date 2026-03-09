---
name: save_note
agents: [search_expert, research]
display:
  progress_message: "Saving note..."
  completed_label: "Saved note"
inputs:
  title:
    type: string
    description: "Title for the note"
  content:
    type: string
    description: "Markdown content of the note"
output_type: string
---

Save a research note to shared memory.

Notes are shared between all agents and persist across sessions. They are your primary way to build up knowledge over time and communicate important findings.

**When to use:**
- You've synthesized an important conclusion from multiple sources
- You've found a key fact worth remembering across questions
- You want to share a finding with other agents working on this research

**Do NOT save:**
- Restatements of the user's question
- Raw search results
- Trivial observations
