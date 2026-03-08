"""Search skill - hybrid document search with RRF fusion."""

import json
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
    document_exists,
    get_chunk_window_by_chunk_id,
    get_document_chunks,
    hybrid_search,
)
from bartleby.write.skills.base import load_tool_doc


def _sanitize_limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_SEARCH_RESULT_LIMIT
    return max(1, min(limit, MAX_SEARCH_RESULT_LIMIT))


def _result_with_body(results: list, max_body_chars: int = 300, ref_registry=None) -> list[Dict]:
    """Return search results with truncated body text included."""
    out = []
    for r in results:
        d = r.to_dict()
        body = d.get("body")
        if body and len(body) > max_body_chars:
            d["body"] = body[:max_body_chars] + "..."
        if ref_registry is not None:
            ref_num = ref_registry.register(d["chunk_id"], d)
            d["ref"] = ref_num
        out.append(d)
    return out


class HybridSearchTool(Tool):
    name = "search_documents"
    description = load_tool_doc("search_documents")
    inputs = {
        "query": {"type": "string", "description": "Search query (keywords or natural language)"},
        "limit": {
            "type": "integer",
            "description": "Maximum results to return (default 3, max 5)",
            "nullable": True,
        },
        "document_id": {
            "type": "string",
            "description": "Optional document ID to search within",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, connection, embedding_model=None, embedding_lock=None,
                 ref_registry=None, reranker=None):
        super().__init__()
        self.connection = connection
        self.embedding_model = embedding_model
        self.embedding_lock = embedding_lock
        self.ref_registry = ref_registry
        self.reranker = reranker

    def forward(self, query: str, limit: int = None, document_id: str = None) -> str:
        safe_limit = _sanitize_limit(limit)
        if document_id and not document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        results = hybrid_search(
            self.connection,
            query,
            self.embedding_model,
            embedding_lock=self.embedding_lock,
            limit=safe_limit,
            document_id=document_id,
            reranker=self.reranker,
        )
        data = _result_with_body(results, ref_registry=self.ref_registry)
        return json.dumps(truncate_result(data, max_tokens=MAX_TOOL_TOKENS), default=str)


class GetChunkWindowTool(Tool):
    name = "get_chunk_window"
    description = load_tool_doc("get_chunk_window")
    inputs = {
        "chunk_id": {"type": "string", "description": "Chunk ID from a search result"},
        "window_radius": {
            "type": "integer",
            "description": "Chunks before/after the anchor (default 3)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, connection, ref_registry=None):
        super().__init__()
        self.connection = connection
        self.ref_registry = ref_registry

    def forward(self, chunk_id: str, window_radius: int = None) -> str:
        if window_radius is None:
            window_radius = DEFAULT_CHUNK_WINDOW_RADIUS
        safe_radius = max(0, min(window_radius, MAX_DOCUMENT_CHUNK_WINDOW // 2))

        window = get_chunk_window_by_chunk_id(self.connection, chunk_id, safe_radius)
        if window is None:
            return json.dumps({"error": f"Chunk {chunk_id} not found"})

        if self.ref_registry is not None:
            for chunk in window.get("chunks", []):
                ref_num = self.ref_registry.register(chunk["chunk_id"], chunk)
                chunk["ref"] = ref_num

        return json.dumps(window, default=str)


class GetFullDocumentTool(Tool):
    name = "get_full_document"
    description = load_tool_doc("get_full_document")
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
