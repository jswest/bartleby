"""Save a research note to shared memory."""

import json
from datetime import date
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills._base import load_skill_meta, slugify

meta = load_skill_meta(__file__)


class SaveNoteTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

    def __init__(self, memory_dir: Path):
        super().__init__()
        self.memory_dir = memory_dir

    def forward(self, title: str, content: str) -> str:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        slug = slugify(title)
        today = date.today().isoformat()
        filename = f"{today}_{slug}.md"
        filepath = self.memory_dir / filename

        note_content = f"# {title}\n\n{content}\n"
        filepath.write_text(note_content, encoding="utf-8")

        return json.dumps({
            "message": f"Saved note: {title}",
            "filename": filename,
        })


def create(context: dict) -> SaveNoteTool:
    return SaveNoteTool(memory_dir=context["memory_dir"])
