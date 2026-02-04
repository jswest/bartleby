"""Base class for skills."""

from smolagents import Tool


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
