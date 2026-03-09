"""Read a window of text around a specific search hit."""

import json

from smolagents import Tool

from bartleby.lib.consts import DEFAULT_CHUNK_WINDOW_RADIUS, MAX_DOCUMENT_CHUNK_WINDOW
from bartleby.write.search import get_chunk_window_by_chunk_id
from bartleby.write.skills._base import load_skill_meta

meta = load_skill_meta(__file__)


class GetChunkWindowTool(Tool):
    name = meta.name
    description = meta.description
    inputs = meta.inputs
    output_type = meta.output_type

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


def create(context: dict) -> GetChunkWindowTool:
    return GetChunkWindowTool(
        connection=context["connection"],
        ref_registry=context.get("ref_registry"),
    )
