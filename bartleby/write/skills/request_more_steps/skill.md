---
name: request_more_steps
agents: [research]
inputs:
  reason:
    type: string
    description: "Brief explanation of why you need more steps and what you still need to do"
  additional_steps:
    type: integer
    description: "Number of additional steps requested (typically 5)"
    nullable: true
output_type: string
---

Request additional steps when you are running low and need more time to complete your research. Call this when you are approaching the step limit and still have important work to do. The user will be asked to approve the extension.
