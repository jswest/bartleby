"""Read all saved research notes from shared memory."""

import json
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills._base import load_skill_meta

meta = load_skill_meta(__file__)


class ReadNotesTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

    def __init__(self, memory_dir: Path):
        super().__init__()
        self.memory_dir = memory_dir

    def forward(self) -> str:
        if not self.memory_dir.exists():
            return json.dumps({"message": "No notes found.", "notes": []})

        note_files = sorted(self.memory_dir.glob("*.md"))
        if not note_files:
            return json.dumps({"message": "No notes found.", "notes": []})

        notes = []
        for f in note_files:
            notes.append({
                "filename": f.name,
                "content": f.read_text(encoding="utf-8")
            })
        return json.dumps({"count": len(notes), "notes": notes})


def create(context: dict) -> ReadNotesTool:
    return ReadNotesTool(memory_dir=context["memory_dir"])
