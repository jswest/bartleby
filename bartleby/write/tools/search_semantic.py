"""Semantic search tool for meaning-based document searching."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from threading import Lock

from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from bartleby.lib.consts import (
    DEFAULT_SEARCH_RESULT_LIMIT,
    MAX_SEARCH_RESULT_LIMIT,
    MAX_TOOL_TOKENS,
)
from bartleby.lib.utils import truncate_result
from bartleby.write.search import semantic_search


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

    def sanitize_limit(limit: Optional[int]) -> int:
        """Clamp search limits to keep tool outputs lean."""
        if limit is None:
            return DEFAULT_SEARCH_RESULT_LIMIT
        return max(1, min(limit, MAX_SEARCH_RESULT_LIMIT))

    def result_metadata(results: List) -> List[Dict[str, Any]]:
        """Strip bulky chunk bodies from search responses."""
        return [r.to_metadata_dict() for r in results]

    @tool
    def search_documents_semantic(
        query: str,
        limit: int = DEFAULT_SEARCH_RESULT_LIMIT,
        document_id: str | None = None
    ) -> List[Dict[str, Any]]:
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
        if before_hook:
            preempt = before_hook("search_documents_semantic")
            if preempt is not None:
                return preempt

        limit = sanitize_limit(limit)
        with embedding_lock:
            results = semantic_search(
                db_path, query, embedding_model, limit, document_id=document_id
            )
        data = result_metadata(results)
        return truncate_result(data, max_tokens=MAX_TOOL_TOKENS)

    return search_documents_semantic
