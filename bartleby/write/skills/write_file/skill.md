---
name: write_file
agents: [research]
inputs:
  filename:
    type: string
    description: "Filename to write (e.g., 'draft.md')"
  content:
    type: string
    description: "Content to write to the file"
output_type: string
---

Write content to a file in the output directory.

Use this to save drafts, reports, data exports, or other artifacts. Provide a simple filename (no paths) and the content to write.

**When to use:**
- The user asks you to save or export something
- You've prepared a report or summary to deliver
- You need to output structured data (CSV, JSON, etc.)

**Note:** Files are saved to the project's book directory.
