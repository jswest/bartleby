"""Step extension skill - allows the agent to request more steps."""

import json

from rich.prompt import Confirm
from rich.console import Console
from smolagents import Tool


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

    def __init__(self, agent_ref: list, console: Console, live_ref: list | None = None):
        super().__init__()
        # agent_ref is a mutable list holding [agent] so we can access it
        # after the agent is created (circular reference workaround)
        self._agent_ref = agent_ref
        self._console = console
        # live_ref holds the active Live display so we can pause it during prompts
        self._live_ref = live_ref or []

    def forward(self, reason: str, additional_steps: int = None) -> str:
        requested = additional_steps if additional_steps is not None else 5

        # Pause the Live display so the prompt renders cleanly
        live = self._live_ref[0] if self._live_ref else None
        if live:
            live.stop()
        self._console.print("")
        self._console.print(
            f"[bold yellow]Agent requests {requested} more steps:[/bold yellow] {reason}"
        )
        approved = Confirm.ask("Allow?", default=False, console=self._console)
        if live:
            live.start()

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
