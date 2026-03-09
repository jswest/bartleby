"""Generate and cache an LLM summary for a document."""

import json

import litellm
from loguru import logger
from smolagents import Tool

from bartleby.read.llm import _parse_summary_response, SUMMARIZE_PROMPT
from bartleby.write.search import (
    document_exists,
    get_document_chunks,
    get_document_summary,
    save_document_summary,
)
from bartleby.lib.utils import safe_dumps
from bartleby.write.skills._base import load_skill_meta

MAX_FIRST_CHUNKS = 15
MAX_SUMMARIZE_INPUT_CHARS = 8000

meta = load_skill_meta(__file__)


class SummarizeDocumentTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

    def __init__(self, connection, model_id=None):
        super().__init__()
        self.connection = connection
        self.model_id = model_id

    def forward(self, document_id: str) -> str:
        if not document_exists(self.connection, document_id):
            return json.dumps({
                "error": "DOCUMENT_NOT_FOUND",
                "message": f"Document '{document_id}' was not found.",
            })

        # Check if already cached
        existing = get_document_summary(self.connection, document_id)
        if existing:
            return safe_dumps({
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

        return safe_dumps({
            "document_id": document_id,
            "source": "summary",
            "title": summary.title,
            "subtitle": summary.subtitle,
            "body": summary.body,
            "note": "Summary generated and cached.",
        })


def create(context: dict) -> SummarizeDocumentTool:
    return SummarizeDocumentTool(
        connection=context["connection"],
        model_id=context.get("model_id"),
    )
