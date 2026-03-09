---
name: write_csv
agents: [research]
display:
  progress_message: "Writing CSV..."
  completed_label: "Wrote CSV"
inputs:
  filename:
    type: string
    description: "Output filename (e.g., 'mentions.csv'). Must end in .csv."
  content:
    type: string
    description: "CSV-formatted content including header row."
output_type: string
---

Write a CSV file to the output directory.

Use this when the user's question calls for structured, tabular data — for example:
- Counting mentions of a term across documents
- Listing which documents discuss a topic
- Comparing data points across sources
- Any question where a spreadsheet would be more useful than prose

Format the content as standard CSV with a header row. Use proper quoting for values containing commas.
