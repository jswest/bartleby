"""File write skill - write files to the output directory."""

import json
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills.base import Skill


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write content to a file in the output directory. "
        "Use this for saving drafts, code snippets, data exports, etc. "
        "Filenames must be simple (no path traversal)."
    )
    inputs = {
        "filename": {"type": "string", "description": "Filename to write (e.g., 'draft.md')"},
        "content": {"type": "string", "description": "Content to write to the file"},
    }
    output_type = "string"

    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir

    def forward(self, filename: str, content: str) -> str:
        # Validate filename - no path traversal
        clean_name = Path(filename).name
        if clean_name != filename or ".." in filename or "/" in filename:
            return json.dumps({
                "error": f"Invalid filename: '{filename}'. Use a simple filename without paths."
            })

        filepath = self.output_dir / clean_name
        filepath.write_text(content, encoding="utf-8")

        return json.dumps({
            "message": f"Wrote {len(content)} characters to {clean_name}",
            "filepath": str(filepath),
        })


class FileWriteSkill(Skill):
    name = "file_write"
    description = "Write files to the output directory"

    def get_tools(self, context: dict) -> list[Tool]:
        # Use the book directory as output
        output_dir = context["findings_dir"].parent
        return [WriteFileTool(output_dir)]
