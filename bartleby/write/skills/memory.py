"""Memory skill - notes and findings management."""

import json
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills.base import Skill


class SaveNoteTool(Tool):
    name = "save_note"
    description = (
        "Save a markdown note to the findings directory. "
        "Use this to record research summaries, intermediate findings, "
        "or any structured notes you want to reference later."
    )
    inputs = {
        "title": {"type": "string", "description": "Title for the note"},
        "content": {"type": "string", "description": "Markdown content of the note"},
    }
    output_type = "string"

    def __init__(self, findings_dir: Path, run_uuid: str):
        super().__init__()
        self.findings_dir = findings_dir
        self.run_uuid = run_uuid
        self._sequence = 0

    def forward(self, title: str, content: str) -> str:
        self._sequence += 1
        filename = f"{self.run_uuid}-note-{self._sequence:02d}.md"
        filepath = self.findings_dir / filename

        note_content = f"# {title}\n\n{content}\n"
        filepath.write_text(note_content, encoding="utf-8")

        return json.dumps({
            "message": f"Saved note: {title}",
            "filename": filename,
            "sequence": self._sequence,
        })


class MemorySkill(Skill):
    name = "memory"
    description = "Notes and research findings management"

    def get_tools(self, context: dict) -> list[Tool]:
        findings_dir = context["findings_dir"]
        run_uuid = context["run_uuid"]

        return [
            SaveNoteTool(findings_dir, run_uuid),
        ]
