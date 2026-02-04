"""Search skill - full-text and semantic document search."""

import json
from threading import Lock
from typing import Dict, Optional

from smolagents import Tool

from bartleby.lib.consts import (
    DEFAULT_CHUNK_WINDOW_RADIUS,
    DEFAULT_SEARCH_RESULT_LIMIT,
    MAX_DOCUMENT_CHUNK_WINDOW,
    MAX_SEARCH_RESULT_LIMIT,
    MAX_TOOL_TOKENS,
)
from bartleby.lib.utils import truncate_result
from bartleby.write.search import (
    count_document_chunks,
    full_text_search,
    get_chunk_window_by_chunk_id,
    get_document_chunks,
    semantic_search,
)
from bartleby.write.skills.base import Skill


def _sanitize_limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_SEARCH_RESULT_LIMIT
    return max(1, min(limit, MAX_SEARCH_RESULT_LIMIT))


def _result_with_body(results: list, max_body_chars: int = 300) -> list[Dict]:
    """Return search results with truncated body text included."""
    out = []
    for r in results:
        d = r.to_dict()
        body = d.get("body")
        if body and len(body) > max_body_chars:
            d["body"] = body[:max_body_chars] + "..."
        out.append(d)
    return out


def _document_exists(connection, document_id: str) -> bool:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
        (document_id,),
    )
    return cursor.fetchone() is not None


class SearchDocumentsFTSTool(Tool):
    name = "search_documents_fts"
    description = (
        "Search documents using full-text search (keyword matching). "
        "Best for: exact phrases, technical terms, specific names. "
        "Example: 'artificial intelligence', 'quarterly revenue', 'John Smith'. "
        "Supports FTS5 syntax (AND, OR, NOT)."
    )
    inputs = {
        "query": {"type": "string", "description": "Search query string"},
        "limit": {
            "type": "integer",
            "description": "Maximum results (default 3, max 5)",
            "nullable": True,
        },
        "document_id": {
            "type": "string",
            "description": "Optional document ID to search within",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def forward(self, query: str, limit: int = None, document_id: str = None) -> str:
        safe_limit = _sanitize_limit(limit)
        if document_id and not _document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        results = full_text_search(
            self.connection, query, safe_limit, document_id=document_id or None
        )
        data = _result_with_body(results)
        return json.dumps(truncate_result(data, max_tokens=MAX_TOOL_TOKENS), default=str)


class SearchDocumentsSemanticTool(Tool):
    name = "search_documents_semantic"
    description = (
        "Search documents using semantic similarity (meaning-based). "
        "Best for: conceptual queries, finding related content, paraphrases. "
        "Example: 'what are the main findings?', 'how does this system work?'"
    )
    inputs = {
        "query": {"type": "string", "description": "Natural language query"},
        "limit": {
            "type": "integer",
            "description": "Maximum results (default 3, max 5)",
            "nullable": True,
        },
        "document_id": {
            "type": "string",
            "description": "Optional document ID to search within",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, connection, embedding_model, embedding_lock: Lock):
        super().__init__()
        self.connection = connection
        self.embedding_model = embedding_model
        self.embedding_lock = embedding_lock

    def forward(self, query: str, limit: int = None, document_id: str = None) -> str:
        safe_limit = _sanitize_limit(limit)
        if document_id and not _document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        with self.embedding_lock:
            results = semantic_search(
                self.connection,
                query,
                self.embedding_model,
                safe_limit,
                document_id=document_id or None,
            )
        data = _result_with_body(results)
        return json.dumps(truncate_result(data, max_tokens=MAX_TOOL_TOKENS), default=str)


class GetChunkWindowTool(Tool):
    name = "get_chunk_window"
    description = (
        "Read a small window of chunks around a specific search hit. "
        "Provide the chunk_id from any search result to grab nearby text "
        "(default ~3 chunks on either side)."
    )
    inputs = {
        "chunk_id": {"type": "string", "description": "Chunk ID from a search result"},
        "window_radius": {
            "type": "integer",
            "description": "Chunks before/after the anchor (default 3)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def forward(self, chunk_id: str, window_radius: int = None) -> str:
        if window_radius is None:
            window_radius = DEFAULT_CHUNK_WINDOW_RADIUS
        safe_radius = max(0, min(window_radius, MAX_DOCUMENT_CHUNK_WINDOW // 2))

        window = get_chunk_window_by_chunk_id(self.connection, chunk_id, safe_radius)
        if window is None:
            return json.dumps({"error": f"Chunk {chunk_id} not found"})

        return json.dumps(window, default=str)


class GetFullDocumentTool(Tool):
    name = "get_full_document"
    description = (
        "Retrieve a window of chunks from a specific document. "
        "Use this to page through long documents. Provide the document_id "
        "from a search result plus optional start_chunk to advance deeper."
    )
    inputs = {
        "document_id": {"type": "string", "description": "Document ID from search results"},
        "start_chunk": {
            "type": "integer",
            "description": "Zero-based index to start from (default 0)",
            "nullable": True,
        },
        "max_chunks": {
            "type": "integer",
            "description": "Max chunks to return (capped at 100)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def forward(self, document_id: str, start_chunk: int = None, max_chunks: int = None) -> str:
        total_chunks = count_document_chunks(self.connection, document_id)
        if total_chunks == 0:
            return json.dumps({"error": f"Document {document_id} not found"})

        safe_start = max(0, start_chunk if start_chunk else 0)
        window_size = max(1, min(max_chunks if max_chunks else MAX_DOCUMENT_CHUNK_WINDOW, MAX_DOCUMENT_CHUNK_WINDOW))

        if safe_start >= total_chunks:
            return json.dumps({
                "error": f"start_chunk {safe_start} is beyond document length ({total_chunks} chunks)",
                "total_chunks": total_chunks,
            })

        chunks = get_document_chunks(
            self.connection,
            document_id,
            start_chunk=safe_start,
            max_chunks=window_size,
        )

        end_chunk = safe_start + len(chunks)
        return json.dumps({
            "document_id": document_id,
            "origin_file_path": chunks[0].origin_file_path if chunks else None,
            "total_chunks": total_chunks,
            "start_chunk": safe_start,
            "returned_chunks": len(chunks),
            "has_more": end_chunk < total_chunks,
            "next_start_chunk": end_chunk if end_chunk < total_chunks else None,
            "chunks": [r.to_dict() for r in chunks],
        }, default=str)


class SearchSkill(Skill):
    name = "search"
    description = "Full-text and semantic document search tools"

    def get_tools(self, context: dict) -> list[Tool]:
        connection = context["connection"]
        embedding_model = context.get("embedding_model")
        embedding_lock = context.get("embedding_lock", Lock())

        tools = [
            SearchDocumentsFTSTool(connection),
            GetChunkWindowTool(connection),
            GetFullDocumentTool(connection),
        ]

        if embedding_model:
            tools.insert(1, SearchDocumentsSemanticTool(connection, embedding_model, embedding_lock))

        return tools
