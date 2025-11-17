"""Semantic search tool for meaning-based document searching."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
from threading import Lock

from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from bartleby.lib.consts import DEFAULT_SEARCH_RESULT_LIMIT, MAX_TOOL_TOKENS
from bartleby.lib.utils import truncate_result
from bartleby.write.search import semantic_search
from bartleby.write.tools.common import (
    sanitize_limit,
    result_metadata,
    with_hook,
    document_exists,
)


def create_search_semantic_tool(
    db_path: Path,
    embedding_model: SentenceTransformer,
    embedding_lock: Lock,
    before_hook: Optional[Callable[[str], Any]] = None,
):
    """
    Create semantic search tool.

    Args:
        db_path: Path to document database
        embedding_model: SentenceTransformer model for embeddings
        embedding_lock: Thread lock for embedding model access
        before_hook: Optional hook called before tool execution

    Returns:
        LangChain tool instance
    """

    @tool
    @with_hook("search_documents_semantic", before_hook)
    def search_documents_semantic(
        query: str,
        limit: int = DEFAULT_SEARCH_RESULT_LIMIT,
        document_id: str = "",
    ) -> Dict[str, Any]:
        """
        Search documents using semantic similarity (meaning-based).

        Best for: Conceptual queries, finding related content, paraphrases
        Example: "what are the main findings?", "how does this system work?"

        Args:
            query: Natural language query
            limit: Maximum number of results (default: 3, max: 5)
            document_id: Optional document ID to search within

        Returns:
            List of semantically similar document chunks with metadata
        """
        safe_limit = sanitize_limit(limit)
        if document_id and not document_exists(db_path, document_id):
            return {
                "error": "DOCUMENT_NOT_FOUND",
                "message": (
                    f"Document '{document_id}' was not found. "
                    "Use a valid document_id from the report/findings or leave this blank."
                ),
            }

        with embedding_lock:
            results = semantic_search(
                db_path,
                query,
                embedding_model,
                safe_limit,
                document_id=document_id or None,
            )
        data = result_metadata(results)
        return truncate_result(data, max_tokens=MAX_TOOL_TOKENS)

    return search_documents_semantic
