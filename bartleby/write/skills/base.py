"""Base class for skills."""

from pathlib import Path

from smolagents import Tool


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


class Skill:
    """Base class for skills. Each skill provides a list of tools."""

    name: str = ""
    description: str = ""

    def get_tools(self, context: dict) -> list[Tool]:
        """
        Return tools for this skill.

        Args:
            context: Shared resources dict with keys:
                - db_path: Path to document database
                - findings_dir: Path to findings directory
                - embedding_model: SentenceTransformer model
                - embedding_lock: Lock for embedding model
                - run_uuid: Unique ID for this run

        Returns:
            List of smolagents Tool instances
        """
        raise NotImplementedError
