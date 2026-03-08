"""Shared utilities for skill tool definitions."""

from pathlib import Path


def load_tool_doc(tool_name: str) -> str:
    """Load tool documentation from a Markdown file.

    Args:
        tool_name: Name of the tool (matches filename without .md)

    Returns:
        Documentation string, or empty string if file not found
    """
    doc_path = Path(__file__).parent / "docs" / f"{tool_name}.md"
    if doc_path.exists():
        return doc_path.read_text(encoding="utf-8").strip()
    return ""
