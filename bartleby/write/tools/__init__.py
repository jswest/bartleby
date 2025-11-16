"""Tool registry for Primary and Search agents."""

from pathlib import Path
from typing import List, Set, Any, Callable, Optional
from threading import Lock

from bartleby.write.memory import TodoList


def get_tools(
    db_path: Path,
    todos_path: Path,
    embedding_model,
    allowed_tools: Set[str],
    findings_dir: Optional[Path] = None,
    run_uuid: Optional[str] = None,
    llm = None,
    token_counter = None,
    logger = None,
    display_callback: Optional[Callable] = None,
    before_hook: Optional[Callable[[str], Any]] = None,
) -> List:
    """
    Build and return requested tools for agents.

    Args:
        db_path: Path to document database
        todos_path: Path to todos.json file
        embedding_model: SentenceTransformer model for semantic search
        allowed_tools: Set of tool names to include
        findings_dir: Directory where findings files are stored (for read_findings)
        run_uuid: UUID of current run (for read_findings, delegate_search)
        llm: Language model (for delegate_search)
        token_counter: Token counter callback (for delegate_search)
        logger: Streaming logger (for delegate_search)
        display_callback: Display update callback (for delegate_search)
        before_hook: Optional hook function called before each tool execution

    Returns:
        List of tool instances
    """
    # Lazy imports to avoid circular dependencies
    from bartleby.write.tools.search_fts import create_search_fts_tool
    from bartleby.write.tools.search_semantic import create_search_semantic_tool
    from bartleby.write.tools.get_full_document import create_get_full_document_tool
    from bartleby.write.tools.get_chunk_window import create_get_chunk_window_tool
    from bartleby.write.tools.manage_todo import create_manage_todo_tool
    from bartleby.write.tools.read_findings import create_read_findings_tool
    from bartleby.write.tools.delegate_search import create_delegate_search_tool

    # Shared state
    todo_list = TodoList(str(todos_path))
    embedding_lock = Lock()

    # Tool factory registry
    registry = {
        "search_documents_fts": lambda: create_search_fts_tool(db_path, before_hook),
        "search_documents_semantic": lambda: create_search_semantic_tool(
            db_path, embedding_model, embedding_lock, before_hook
        ),
        "get_full_document": lambda: create_get_full_document_tool(db_path, before_hook),
        "get_chunk_window": lambda: create_get_chunk_window_tool(db_path, before_hook),
        "manage_todo_tool": lambda: create_manage_todo_tool(todo_list, before_hook),
        "read_findings": lambda: create_read_findings_tool(findings_dir, run_uuid),
        "delegate_search": lambda: create_delegate_search_tool(
            llm=llm,
            db_path=db_path,
            findings_dir=findings_dir,
            todos_path=todos_path,
            embedding_model=embedding_model,
            run_uuid=run_uuid,
            token_counter=token_counter,
            logger=logger,
            display_callback=display_callback,
            todo_list=todo_list,
        ),
    }

    # Build requested tools
    tools = []
    for tool_name in allowed_tools:
        if tool_name in registry:
            tools.append(registry[tool_name]())
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    return tools
