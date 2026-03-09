"""Memory skill - curated research notes shared across agents and sessions."""

import json
import re
from datetime import date
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills.base import load_tool_doc


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe kebab-case slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


class ReadNotesTool(Tool):
    name = "read_notes"
    description = (
        "Read all saved research notes. Notes are shared between you and other "
        "agents, and persist across sessions. Use this to review what has been "
        "discovered so far."
    )
    inputs = {}
    output_type = "string"

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


class SaveNoteTool(Tool):
    name = "save_note"
    description = load_tool_doc("save_note")
    inputs = {
        "title": {"type": "string", "description": "Title for the note"},
        "content": {"type": "string", "description": "Markdown content of the note"},
    }
    output_type = "string"

    def __init__(self, memory_dir: Path):
        super().__init__()
        self.memory_dir = memory_dir

    def forward(self, title: str, content: str) -> str:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        slug = _slugify(title)
        today = date.today().isoformat()
        filename = f"{today}_{slug}.md"
        filepath = self.memory_dir / filename

        note_content = f"# {title}\n\n{content}\n"
        filepath.write_text(note_content, encoding="utf-8")

        return json.dumps({
            "message": f"Saved note: {title}",
            "filename": filename,
        })
