"""Reference registry for tracking numbered source citations."""


class ReferenceRegistry:
    """Assigns 1-based ref numbers to chunks, deduplicating by chunk_id."""

    def __init__(self):
        self._refs = []  # list of metadata dicts, index = ref_num - 1
        self._chunk_id_to_ref = {}  # chunk_id -> ref_num

    def register(self, chunk_id: str, metadata: dict) -> int:
        """Register a chunk and return its ref number.

        If chunk_id is already registered, returns the existing number.
        """
        if chunk_id in self._chunk_id_to_ref:
            return self._chunk_id_to_ref[chunk_id]

        ref_num = len(self._refs) + 1
        entry = {
            "ref": ref_num,
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id"),
            "origin_file_path": metadata.get("origin_file_path"),
            "page_number": metadata.get("page_number"),
            "section_heading": metadata.get("section_heading"),
            "chunk_index": metadata.get("chunk_index"),
        }
        self._refs.append(entry)
        self._chunk_id_to_ref[chunk_id] = ref_num
        return ref_num

    def get(self, ref_num: int) -> dict | None:
        """Look up a ref by number. Returns None if out of range."""
        if 1 <= ref_num <= len(self._refs):
            return self._refs[ref_num - 1]
        return None

    def all_refs(self) -> list[dict]:
        """Return all registered refs."""
        return list(self._refs)

    def clear(self):
        """Reset the registry for a new question."""
        self._refs.clear()
        self._chunk_id_to_ref.clear()
