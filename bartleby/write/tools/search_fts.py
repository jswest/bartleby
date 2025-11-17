"""Full-text search tool for document searching."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from langchain_core.tools import tool

from bartleby.lib.consts import (
    DEFAULT_SEARCH_RESULT_LIMIT,
    MAX_SEARCH_RESULT_LIMIT,
    MAX_TOOL_TOKENS,
)
from bartleby.lib.utils import truncate_result
from bartleby.write.search import full_text_search


def create_search_fts_tool(
    db_path: Path,
    before_hook: Optional[Callable[[str], Any]] = None,
):
    """
    Create full-text search tool.

    Args:
        db_path: Path to document database
        before_hook: Optional hook called before tool execution

    Returns:
        LangChain tool instance
    """

    def sanitize_limit(limit: Optional[int]) -> int:
        """Clamp search limits to keep tool outputs lean."""
        if limit is None:
            return DEFAULT_SEARCH_RESULT_LIMIT
        return max(1, min(limit, MAX_SEARCH_RESULT_LIMIT))

    def result_metadata(results: List) -> List[Dict[str, Any]]:
        """Strip bulky chunk bodies from search responses."""
        return [r.to_metadata_dict() for r in results]

    @tool
    def search_documents_fts(
        query: str,
        limit: int = DEFAULT_SEARCH_RESULT_LIMIT,
        document_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search documents using full-text search (keyword matching).

        Best for: Exact phrases, technical terms, specific names
        Example: "artificial intelligence", "quarterly revenue", "John Smith"

        Args:
            query: Search query string (supports FTS5 syntax like AND, OR, NOT)
            limit: Maximum number of results (default: 3, max: 5)
            document_id: Optional document ID to search within

        Returns:
            List of matching document chunks with metadata
        """
        if before_hook:
            preempt = before_hook("search_documents_fts")
            if preempt is not None:
                return preempt

        limit = sanitize_limit(limit)
        results = full_text_search(
            db_path, query, limit, document_id=document_id or None
        )
        data = result_metadata(results)
        return truncate_result(data, max_tokens=MAX_TOOL_TOKENS)

    return search_documents_fts
