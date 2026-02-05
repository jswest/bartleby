"""Memory skill - notes and findings management."""

import json
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills.base import Skill, load_tool_doc


class ReadNotesTool(Tool):
    name = "read_notes"
    description = "Read all saved research notes from previous questions. Use this when the user asks about prior research, what they've asked before, or what's been found so far."
    inputs = {}
    output_type = "string"

    def __init__(self, findings_dir: Path):
        super().__init__()
        self.findings_dir = findings_dir

    def forward(self) -> str:
        if not self.findings_dir.exists():
            return json.dumps({"message": "No notes found.", "notes": []})

        note_files = sorted(self.findings_dir.glob("*.md"))
        if not note_files:
            return json.dumps({"message": "No notes found.", "notes": []})

        notes = []
        for f in note_files:
            notes.append({
                "filename": f.name,
                "content": f.read_text(encoding="utf-8")
            })
        return json.dumps({"count": len(notes), "notes": notes})


class SaveNoteTool(Tool):
    name = "save_note"
    description = load_tool_doc("save_note")
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
            ReadNotesTool(findings_dir),
            SaveNoteTool(findings_dir, run_uuid),
        ]
