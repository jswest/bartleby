"""Allow the agent to request more steps from the user."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from smolagents import Tool

from bartleby.write.skills._base import load_skill_meta

if TYPE_CHECKING:
    from bartleby.write.renderer import CliRenderer

meta = load_skill_meta(__file__)


class RequestMoreStepsTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

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


def create(context: dict) -> RequestMoreStepsTool:
    return RequestMoreStepsTool(
        agent_ref=context["agent_ref"],
        renderer=context["renderer"],
    )
