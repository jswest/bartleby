"""LangGraph-compatible tools for AI agents to search documents."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from bartleby.lib.consts import (
    EMBEDDING_MODEL,
    MAX_TOOL_TOKENS,
    MAX_DOCUMENT_CHUNK_WINDOW,
    DEFAULT_SEARCH_RESULT_LIMIT,
    MAX_SEARCH_RESULT_LIMIT,
    DEFAULT_CHUNK_WINDOW_RADIUS,
)
from bartleby.lib.utils import load_config, load_llm_from_config, truncate_result
from bartleby.write.memory import TodoList, read_scratchpad, append_to_scratchpad
from bartleby.write.search import (
    full_text_search,
    semantic_search,
    get_document_chunks,
    count_document_chunks,
    get_chunk_window_by_chunk_id,
)


class DocumentSearchTools:
    """
    Collection of search tools for LangGraph agents.

    Usage:
        tools = DocumentSearchTools(db_path="path/to/db/bartleby.db")
        agent_tools = tools.get_tools()
    """

    def __init__(
        self,
        db_path: str | Path,
        scratchpad_path: str | Path,
        todos_path: str | Path,
        embedding_model: SentenceTransformer = None,
    ):
        self.db_path = Path(db_path)
        self.scratchpad_path = Path(scratchpad_path)
        self.todos_path = Path(todos_path)
        self._embedding_model = embedding_model
        self.todo_list = TodoList(self.todos_path)
        self._embedding_lock = Lock()  # Thread-safe access to embedding model

    @property
    def llm(self) -> Optional[BaseLanguageModel]:
        """Lazy-load LLM from config if not provided."""
        config = load_config()
        if self._llm is None:
            self._llm = load_llm_from_config(config)
        return self._llm

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def get_tools(self) -> List:
        """
        Get all search tools as a list for LangGraph.

        Returns:
            List of LangChain tool objects (logging happens via callback handler)
        """
        # Create closure-based tools that capture self
        db_path = self.db_path
        embedding_model_getter = lambda: self.embedding_model
        embedding_lock = self._embedding_lock
        scratchpad_path = self.scratchpad_path
        todo_list = self.todo_list

        def sanitize_limit(limit: Optional[int]) -> int:
            """Clamp search limits to keep tool outputs lean."""
            if limit is None:
                return DEFAULT_SEARCH_RESULT_LIMIT
            return max(1, min(limit, MAX_SEARCH_RESULT_LIMIT))

        def result_metadata(results: List) -> List[Dict[str, Any]]:
            """Strip bulky chunk bodies from search responses."""
            return [r.to_metadata_dict() for r in results]

        @tool
        def search_documents_fts(query: str, limit: int = DEFAULT_SEARCH_RESULT_LIMIT) -> List[Dict[str, Any]]:
            """
            Search documents using full-text search (keyword matching).

            Best for: Exact phrases, technical terms, specific names
            Example: "artificial intelligence", "quarterly revenue", "John Smith"

            Args:
                query: Search query string (supports FTS5 syntax like AND, OR, NOT)
                limit: Maximum number of results (default: 10)

            Returns:
                List of matching document chunks with metadata
            """
            limit = sanitize_limit(limit)
            results = full_text_search(db_path, query, limit)
            data = result_metadata(results)
            return truncate_result(data, max_tokens=MAX_TOOL_TOKENS)

        @tool
        def search_documents_semantic(query: str, limit: int = DEFAULT_SEARCH_RESULT_LIMIT) -> List[Dict[str, Any]]:
            """
            Search documents using semantic similarity (meaning-based).

            Best for: Conceptual queries, finding related content, paraphrases
            Example: "what are the main findings?", "how does this system work?"

            Args:
                query: Natural language query
                limit: Maximum number of results (default: 10)

            Returns:
                List of semantically similar document chunks with metadata
            """
            limit = sanitize_limit(limit)
            with embedding_lock:
                results = semantic_search(db_path, query, embedding_model_getter(), limit)
            data = result_metadata(results)
            return truncate_result(data, max_tokens=MAX_TOOL_TOKENS)

        @tool
        def get_full_document(
            document_id: str,
            start_chunk: int = 0,
            max_chunks: int = MAX_DOCUMENT_CHUNK_WINDOW,
        ) -> Dict[str, Any]:
            """
            Retrieve a window of chunks from a specific document.

            Use this to page through long documents in manageable slices. Provide the document_id
            from a search result plus optional start_chunk to advance deeper into the file.

            Args:
                document_id: The document ID (from search results)
                start_chunk: Zero-based index to start reading from (default 0)
                max_chunks: Maximum number of chunks to return (capped to avoid token blowups)

            Returns:
                Dictionary with metadata, window info, and the requested chunk slice
            """
            total_chunks = count_document_chunks(db_path, document_id)
            if total_chunks == 0:
                return {"error": f"Document {document_id} not found"}

            safe_start = max(0, start_chunk or 0)
            window_size = max(1, min(max_chunks or MAX_DOCUMENT_CHUNK_WINDOW, MAX_DOCUMENT_CHUNK_WINDOW))

            if safe_start >= total_chunks:
                return {
                    "error": f"start_chunk {safe_start} is beyond document length ({total_chunks} chunks)",
                    "total_chunks": total_chunks,
                }

            chunks = get_document_chunks(
                db_path,
                document_id,
                start_chunk=safe_start,
                max_chunks=window_size,
            )

            return {
                "document_id": document_id,
                "origin_file_path": chunks[0].origin_file_path if chunks else None,
                "total_chunks": total_chunks,
                "start_chunk": safe_start,
                "returned_chunks": len(chunks),
                "max_chunks": window_size,
                "has_more": (safe_start + len(chunks)) < total_chunks,
                "next_start_chunk": safe_start + len(chunks) if (safe_start + len(chunks)) < total_chunks else None,
                "chunks": [r.to_dict() for r in chunks],
            }

        @tool
        def get_chunk_window(
            chunk_id: str,
            window_radius: int = DEFAULT_CHUNK_WINDOW_RADIUS,
        ) -> Dict[str, Any]:
            """
            Quickly read a small window of chunks around a specific search hit.

            Provide the chunk_id from any search result and this tool will grab a narrow window
            of nearby text (default ~3 chunks on either side) so you can jump directly to the
            relevant passage without scanning from the start of the document.

            Args:
                chunk_id: Chunk ID returned by a search tool
                window_radius: Number of chunks to include before/after the anchor (default 3)
            """
            safe_radius = max(0, min(window_radius, MAX_DOCUMENT_CHUNK_WINDOW // 2))

            window = get_chunk_window_by_chunk_id(db_path, chunk_id, safe_radius)
            if window is None:
                return {"error": f"Chunk {chunk_id} not found"}

            return window

        @tool
        def read_scratchpad_tool() -> str:
            """
            Read the contents of your scratchpad.

            The scratchpad is persistent storage for notes across your work session.
            Use it to remember important information, track insights, or plan your work.

            Returns:
                Contents of the scratchpad
            """
            return read_scratchpad(scratchpad_path)

        @tool
        def append_to_scratchpad_tool(content: str) -> str:
            """
            Add notes to your scratchpad.

            The scratchpad is persistent storage where you can write notes, reminders,
            or any information you want to remember. Each entry is timestamped.

            Args:
                content: The content to add to your scratchpad

            Returns:
                Confirmation message
            """
            return append_to_scratchpad(scratchpad_path, content)

        @tool
        def manage_todo_tool(action: str, task: str = "", status: str = "") -> Dict[str, Any]:
            """
            Manage your todo list in one place.

            Args:
                action: 'add', 'update', or 'list'
                task: Description of the task (required for add/update)
                status: New status when action='update' ('pending', 'active', 'complete')

            Returns:
                Dictionary with the operation result and current todo info
            """
            action_normalized = (action or "").strip().lower()

            if action_normalized == "add":
                if not task.strip():
                    return {"error": "Task description is required when action='add'."}
                result = todo_list.add_todo(task)
                return {"message": result.get("message"), "todo": result.get("todo"), "total_todos": result.get("total_todos")}

            if action_normalized == "update":
                if not task.strip():
                    return {"error": "Task description is required when action='update'."}
                if status.strip().lower() not in {"pending", "active", "complete"}:
                    return {"error": "Status must be 'pending', 'active', or 'complete' when action='update'."}
                return todo_list.update_todo_status(task, status.lower())

            if action_normalized in {"list", "get"}:
                todos = todo_list.get_todos()
                return {"todos": todos}

            return {"error": "Invalid action. Use 'add', 'update', or 'list'."}

        return [
            search_documents_fts,
            search_documents_semantic,
            get_full_document,
            get_chunk_window,
            read_scratchpad_tool,
            append_to_scratchpad_tool,
            manage_todo_tool,
        ]
