"""Library skill - document listing and summarization."""

import json
from typing import Optional

import litellm
from loguru import logger
from smolagents import Tool

from bartleby.lib.consts import MAX_TOOL_TOKENS
from bartleby.lib.utils import truncate_result
from bartleby.read.llm import DocumentSummary, _parse_summary_response, SUMMARIZE_PROMPT
from bartleby.write.search import (
    count_document_chunks,
    get_document_chunks,
    get_document_summary,
    list_all_documents,
    save_document_summary,
)
from bartleby.write.skills.base import Skill

MAX_FIRST_CHUNKS = 15
MAX_SUMMARIZE_INPUT_CHARS = 8000


def _document_exists(connection, document_id: str) -> bool:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
        (document_id,),
    )
    return cursor.fetchone() is not None


class ListDocumentsTool(Tool):
    name = "list_documents"
    description = (
        "List all documents in the database with metadata. "
        "Returns document IDs, filenames, page counts, chunk counts, "
        "titles, and whether a summary exists."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def forward(self) -> str:
        docs = list_all_documents(self.connection)
        if not docs:
            return json.dumps({"message": "No documents found in the database."})
        return json.dumps(
            truncate_result(docs, max_tokens=MAX_TOOL_TOKENS),
            default=str,
        )


class GetDocumentSummaryTool(Tool):
    name = "get_document_summary"
    description = (
        "Get a summary of a specific document. Returns the cached summary "
        "if available, otherwise returns the first chunks as an approximation. "
        "Check the 'source' field to know which type you received."
    )
    inputs = {
        "document_id": {
            "type": "string",
            "description": "Document ID from list_documents or search results",
        },
    }
    output_type = "string"

    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def forward(self, document_id: str) -> str:
        if not _document_exists(self.connection, document_id):
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
            return json.dumps(
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
        return json.dumps(
            truncate_result(result, max_tokens=MAX_TOOL_TOKENS),
            default=str,
        )


class SummarizeDocumentTool(Tool):
    name = "summarize_document"
    description = (
        "Generate and cache a proper LLM summary for a document. "
        "Use this only when get_document_summary returns a first-chunk "
        "approximation and you need a better understanding of the document."
    )
    inputs = {
        "document_id": {
            "type": "string",
            "description": "Document ID from list_documents or search results",
        },
    }
    output_type = "string"

    def __init__(self, connection, model_id: Optional[str]):
        super().__init__()
        self.connection = connection
        self.model_id = model_id

    def forward(self, document_id: str) -> str:
        if not _document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        # Check if already cached
        existing = get_document_summary(self.connection, document_id)
        if existing:
            return json.dumps({
                "document_id": document_id,
                "source": "summary",
                "title": existing["title"],
                "subtitle": existing["subtitle"],
                "body": existing["body"],
                "note": "Summary was already cached.",
            })

        if not self.model_id:
            return json.dumps({
                "error": "NO_LLM_CONFIGURED",
                "message": "No LLM is configured. Cannot generate summaries.",
            })

        # Gather source text from first chunks
        chunks = get_document_chunks(
            self.connection,
            document_id,
            start_chunk=0,
            max_chunks=MAX_FIRST_CHUNKS,
        )
        if not chunks:
            return json.dumps({
                "error": "NO_CONTENT",
                "message": "No chunks found for this document.",
            })

        source_text = "\n".join(c.body for c in chunks)
        if len(source_text) > MAX_SUMMARIZE_INPUT_CHARS:
            source_text = source_text[:MAX_SUMMARIZE_INPUT_CHARS]

        # Generate summary via LLM
        prompt = SUMMARIZE_PROMPT.format(text=source_text)
        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                return json.dumps({
                    "error": "EMPTY_RESPONSE",
                    "message": "LLM returned an empty response.",
                })

            summary = _parse_summary_response(raw)
        except Exception as e:
            logger.error(f"On-demand summarization failed ({type(e).__name__}): {e}")
            return json.dumps({
                "error": "SUMMARIZATION_FAILED",
                "message": f"Failed to generate summary: {e}",
            })

        # Cache the result
        save_document_summary(
            self.connection,
            document_id,
            summary.title,
            summary.subtitle,
            summary.body,
        )

        return json.dumps({
            "document_id": document_id,
            "source": "summary",
            "title": summary.title,
            "subtitle": summary.subtitle,
            "body": summary.body,
            "note": "Summary generated and cached.",
        })


class LibrarySkill(Skill):
    name = "library"
    description = "Document listing and summarization tools"

    def get_tools(self, context: dict) -> list[Tool]:
        connection = context["connection"]
        model_id = context.get("model_id")

        # Ensure summaries table exists (lazy migration for old databases)
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                document_id TEXT PRIMARY KEY REFERENCES documents(document_id),
                title TEXT NOT NULL,
                subtitle TEXT,
                body TEXT NOT NULL
            )
        """)

        return [
            ListDocumentsTool(connection),
            GetDocumentSummaryTool(connection),
            SummarizeDocumentTool(connection, model_id),
        ]
