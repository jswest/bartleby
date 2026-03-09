"""Write CSV content to a file in the output directory."""

import json
from pathlib import Path

from smolagents import Tool

from bartleby.write.skills._base import load_skill_meta

meta = load_skill_meta(__file__)


class WriteCsvTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

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

        # Enforce .csv extension
        if not clean_name.lower().endswith(".csv"):
            return json.dumps({
                "error": f"Filename must end in .csv, got: '{filename}'"
            })

        filepath = self.output_dir / clean_name
        filepath.write_text(content, encoding="utf-8")

        return json.dumps({
            "message": f"Wrote {len(content)} characters to {clean_name}",
            "filepath": str(filepath),
        })


def create(context: dict) -> WriteCsvTool:
    return WriteCsvTool(output_dir=context["book_dir"])
