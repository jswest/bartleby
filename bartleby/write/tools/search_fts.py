"""Full-text search tool for document searching."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable

from langchain_core.tools import tool

from bartleby.lib.consts import DEFAULT_SEARCH_RESULT_LIMIT, MAX_TOOL_TOKENS
from bartleby.lib.utils import truncate_result
from bartleby.write.search import full_text_search
from bartleby.write.tools.common import (
    sanitize_limit,
    result_metadata,
    with_hook,
    document_exists,
)


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

    @tool
    @with_hook("search_documents_fts", before_hook)
    def search_documents_fts(
        query: str,
        limit: int = DEFAULT_SEARCH_RESULT_LIMIT,
        document_id: str = "",
    ) -> Dict[str, Any]:
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
        safe_limit = sanitize_limit(limit)
        if document_id and not document_exists(db_path, document_id):
            return {
                "error": "DOCUMENT_NOT_FOUND",
                "message": (
                    f"Document '{document_id}' was not found. "
                    "Use a valid document_id from the report/findings or omit this parameter."
                ),
            }

        results = full_text_search(
            db_path, query, safe_limit, document_id=document_id or None
        )
        data = result_metadata(results)
        return truncate_result(data, max_tokens=MAX_TOOL_TOKENS)

    return search_documents_fts
