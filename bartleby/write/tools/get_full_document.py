"""Tool for retrieving full documents or document windows."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable

from langchain_core.tools import tool

from bartleby.lib.consts import MAX_DOCUMENT_CHUNK_WINDOW
from bartleby.write.search import get_document_chunks, count_document_chunks


def create_get_full_document_tool(
    db_path: Path,
    before_hook: Optional[Callable[[str], Any]] = None,
):
    """
    Create full document retrieval tool.

    Args:
        db_path: Path to document database
        before_hook: Optional hook called before tool execution

    Returns:
        LangChain tool instance
    """

    @tool
    def get_full_document(
        document_id: str,
        start_chunk: int = 0,
        max_chunks: int = MAX_DOCUMENT_CHUNK_WINDOW,
    ) -> Dict[str, Any]:
        """
        Retrieve a window of chunks from a specific document.

        Use this to page through long documents in manageable slices. Provide the document_id
        from a search result plus optional start_chunk to advance deeper into the file.

        Args:
            document_id: The document ID (from search results)
            start_chunk: Zero-based index to start reading from (default 0)
            max_chunks: Maximum number of chunks to return (capped to avoid token blowups)

        Returns:
            Dictionary with metadata, window info, and the requested chunk slice
        """
        if before_hook:
            preempt = before_hook("get_full_document")
            if preempt is not None:
                return preempt

        total_chunks = count_document_chunks(db_path, document_id)
        if total_chunks == 0:
            return {"error": f"Document {document_id} not found"}

        safe_start = max(0, start_chunk or 0)
        window_size = max(1, min(max_chunks or MAX_DOCUMENT_CHUNK_WINDOW, MAX_DOCUMENT_CHUNK_WINDOW))

        if safe_start >= total_chunks:
            return {
                "error": f"start_chunk {safe_start} is beyond document length ({total_chunks} chunks)",
                "total_chunks": total_chunks,
            }

        chunks = get_document_chunks(
            db_path,
            document_id,
            start_chunk=safe_start,
            max_chunks=window_size,
        )

        return {
            "document_id": document_id,
            "origin_file_path": chunks[0].origin_file_path if chunks else None,
            "total_chunks": total_chunks,
            "start_chunk": safe_start,
            "returned_chunks": len(chunks),
            "max_chunks": window_size,
            "has_more": (safe_start + len(chunks)) < total_chunks,
            "next_start_chunk": safe_start + len(chunks) if (safe_start + len(chunks)) < total_chunks else None,
            "chunks": [r.to_dict() for r in chunks],
        }

    return get_full_document
