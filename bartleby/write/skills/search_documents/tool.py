"""Hybrid document search with RRF fusion."""

import json
from typing import Dict, Optional

from smolagents import Tool

from bartleby.lib.consts import (
    DEFAULT_SEARCH_RESULT_LIMIT,
    MAX_SEARCH_RESULT_LIMIT,
    MAX_TOOL_TOKENS,
)
from bartleby.lib.utils import truncate_result
from bartleby.write.search import document_exists, full_text_search, hybrid_search
from bartleby.write.skills._base import load_skill_meta

meta = load_skill_meta(__file__)


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
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

    def __init__(self, connection, embedding_model=None, embedding_lock=None,
                 ref_registry=None, reranker=None):
        super().__init__()
        self.connection = connection
        self.embedding_model = embedding_model
        self.embedding_lock = embedding_lock
        self.ref_registry = ref_registry
        self.reranker = reranker

    def forward(self, query: str, limit: int = None, document_id: str = None,
                exact_match: bool = None) -> str:
        safe_limit = _sanitize_limit(limit)
        if document_id and not document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        if exact_match:
            # Exact phrase search: FTS5 only, wrap query in quotes
            # Strip any existing quotes to avoid malformed FTS5 syntax
            phrase_query = f'"{query.replace(chr(34), "")}"'
            results = full_text_search(
                self.connection, phrase_query, safe_limit, document_id=document_id,
            )
        else:
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


def create(context: dict) -> HybridSearchTool:
    return HybridSearchTool(
        connection=context["connection"],
        embedding_model=context.get("embedding_model"),
        embedding_lock=context.get("embedding_lock"),
        ref_registry=context.get("ref_registry"),
        reranker=context.get("reranker"),
    )
