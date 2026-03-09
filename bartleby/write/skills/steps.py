"""Step extension skill - allows the agent to request more steps."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from smolagents import Tool

if TYPE_CHECKING:
    from bartleby.write.renderer import CliRenderer


class RequestMoreStepsTool(Tool):
    name = "request_more_steps"
    description = (
        "Request additional steps when you are running low and need more time "
        "to complete your research. Call this when you are approaching the step "
        "limit and still have important work to do. The user will be asked to "
        "approve the extension."
    )
    inputs = {
        "reason": {
            "type": "string",
            "description": "Brief explanation of why you need more steps and what you still need to do",
        },
        "additional_steps": {
            "type": "integer",
            "description": "Number of additional steps requested (typically 5)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, agent_ref: list, renderer: CliRenderer):
        super().__init__()
        self._agent_ref = agent_ref
        self._renderer = renderer

    def forward(self, reason: str, additional_steps: int = None) -> str:
        requested = additional_steps if additional_steps is not None else 5

        self._renderer.pause_live()
        approved = self._renderer.prompt_step_extension(reason, requested)
        self._renderer.resume_live()

        if approved:
            agent = self._agent_ref[0] if self._agent_ref else None
            if agent is None:
                return json.dumps({
                    "approved": False,
                    "message": "Internal error: could not extend steps.",
                })
            agent.max_steps += requested
            return json.dumps({
                "approved": True,
                "additional_steps": requested,
                "message": f"Approved. You now have {requested} more steps.",
            })
        else:
            return json.dumps({
                "approved": False,
                "message": "Denied. Wrap up your answer with what you have so far.",
            })
