"""Get a cached summary for a document."""

import json

from smolagents import Tool

from bartleby.lib.consts import MAX_TOOL_TOKENS
from bartleby.lib.utils import safe_dumps, truncate_result
from bartleby.write.search import (
    count_document_chunks,
    document_exists,
    get_document_chunks,
    get_document_summary,
)
from bartleby.write.skills._base import load_skill_meta

MAX_FIRST_CHUNKS = 15

meta = load_skill_meta(__file__)


class GetDocumentSummaryTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def forward(self, document_id: str) -> str:
        if not document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        # Try cached summary first
        summary = get_document_summary(self.connection, document_id)
        if summary:
            result = {
                "document_id": document_id,
                "source": "summary",
                "title": summary["title"],
                "subtitle": summary["subtitle"],
                "body": summary["body"],
            }
            return safe_dumps(
                truncate_result(result, max_tokens=MAX_TOOL_TOKENS),
                default=str,
            )

        # Fall back to first N chunks
        chunks = get_document_chunks(
            self.connection,
            document_id,
            start_chunk=0,
            max_chunks=MAX_FIRST_CHUNKS,
        )
        if not chunks:
            return json.dumps({
                "document_id": document_id,
                "source": "first_chunks",
                "note": "No chunks found for this document.",
                "body": "",
            })

        body = "\n".join(c.body for c in chunks)
        total = count_document_chunks(self.connection, document_id)

        result = {
            "document_id": document_id,
            "source": "first_chunks",
            "note": (
                "This is an approximation from the first chunks of the document, "
                "not a true summary. Use summarize_document to generate a proper summary."
            ),
            "chunks_shown": len(chunks),
            "total_chunks": total,
            "body": body,
        }
        return safe_dumps(
            truncate_result(result, max_tokens=MAX_TOOL_TOKENS),
            default=str,
        )


def create(context: dict) -> GetDocumentSummaryTool:
    return GetDocumentSummaryTool(connection=context["connection"])
