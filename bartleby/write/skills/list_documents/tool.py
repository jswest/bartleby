"""List all documents in the corpus."""

import json

from smolagents import Tool

from bartleby.lib.consts import MAX_TOOL_TOKENS
from bartleby.lib.utils import truncate_result
from bartleby.write.search import list_all_documents
from bartleby.write.skills._base import load_skill_meta

meta = load_skill_meta(__file__)


class ListDocumentsTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

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


def create(context: dict) -> ListDocumentsTool:
    return ListDocumentsTool(connection=context["connection"])
