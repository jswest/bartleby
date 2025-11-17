"""Tool for retrieving chunk windows around search results."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable

from langchain_core.tools import tool

from bartleby.lib.consts import DEFAULT_CHUNK_WINDOW_RADIUS, MAX_DOCUMENT_CHUNK_WINDOW
from bartleby.write.search import get_chunk_window_by_chunk_id
from bartleby.write.tools.common import with_hook


def create_get_chunk_window_tool(
    db_path: Path,
    before_hook: Optional[Callable[[str], Any]] = None,
):
    """
    Create chunk window retrieval tool.

    Args:
        db_path: Path to document database
        before_hook: Optional hook called before tool execution

    Returns:
        LangChain tool instance
    """

    @tool
    @with_hook("get_chunk_window", before_hook)
    def get_chunk_window(
        chunk_id: str,
        window_radius: int = DEFAULT_CHUNK_WINDOW_RADIUS,
    ) -> Dict[str, Any]:
        """
        Quickly read a small window of chunks around a specific search hit.

        Provide the chunk_id from any search result and this tool will grab a narrow window
        of nearby text (default ~3 chunks on either side) so you can jump directly to the
        relevant passage without scanning from the start of the document.

        Args:
            chunk_id: Chunk ID returned by a search tool
            window_radius: Number of chunks to include before/after the anchor (default 3)

        Returns:
            Dictionary with chunk window and metadata
        """
        safe_radius = max(0, min(window_radius, MAX_DOCUMENT_CHUNK_WINDOW // 2))

        window = get_chunk_window_by_chunk_id(db_path, chunk_id, safe_radius)
        if window is None:
            return {"error": f"Chunk {chunk_id} not found"}

        return window

    return get_chunk_window
