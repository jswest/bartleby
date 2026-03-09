"""Retrieve chunks from a specific document with pagination."""

import json

from smolagents import Tool

from bartleby.lib.consts import MAX_DOCUMENT_CHUNK_WINDOW
from bartleby.write.search import count_document_chunks, get_document_chunks
from bartleby.write.skills._base import load_skill_meta

meta = load_skill_meta(__file__)


class GetFullDocumentTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

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


def create(context: dict) -> GetFullDocumentTool:
    return GetFullDocumentTool(connection=context["connection"])
