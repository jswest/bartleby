"""Tool for delegating research tasks to the Search Agent."""

from pathlib import Path
from typing import Callable, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from bartleby.lib.consts import DEFAULT_MAX_SEARCH_OPERATIONS
from bartleby.write.memory import TodoList


def create_delegate_search_tool(
    llm: BaseLanguageModel,
    db_path: Path,
    findings_dir: Path,
    todos_path: Path,
    embedding_model,
    run_uuid: str,
    token_counter=None,
    logger=None,
    display_callback: Optional[Callable] = None,
    todo_list: TodoList = None,
    max_search_operations: int = DEFAULT_MAX_SEARCH_OPERATIONS,
):
    """
    Create delegation tool for Primary Agent.

    Args:
        llm: Language model
        db_path: Path to document database
        findings_dir: Directory for findings files
        todos_path: Path to todos.json
        embedding_model: SentenceTransformer model
        run_uuid: UUID of current run
        token_counter: Token counter callback
        logger: Streaming logger
        display_callback: Display update callback
        todo_list: TodoList instance

    Returns:
        LangChain tool instance
    """
    # Lazy import to avoid circular dependency
    from bartleby.write.search_agent import run_search_agent

    # Sequence counter for findings files
    sequence_counter = {"count": 0}

    class DelegateSearchInput(BaseModel):
        task: str = Field(
            ...,
            description="One or more newline-separated todo descriptions to delegate (max 3).",
        )
        details: str = Field(
            default="",
            description="Optional extra context applied to every delegated sub-task.",
        )

    def _parse_tasks(task_block: str) -> list[str]:
        if not task_block:
            return []
        lines = [line.strip(" •-\t") for line in task_block.splitlines() if line.strip()]
        return lines[:3] if lines else [task_block.strip()][:3]

    def _ensure_todo(task_description: str) -> dict:
        """Find existing todo (case-insensitive) or create a new pending one."""
        if not todo_list:
            return {"task": task_description, "status": "pending"}

        existing = todo_list.find_exact(task_description)
        if existing:
            return existing

        result = todo_list.add_todo(task_description)
        return result.get("todo", {"task": task_description, "status": "pending"})

    def _mark_status(task_description: str, status: str) -> dict:
        """Update todo status, returning the resulting record."""
        if not todo_list:
            return {"task": task_description, "status": status}

        if status == "active":
            result = todo_list.set_active_task(task_description)
        else:
            result = todo_list.update_todo_status_exact(task_description, status)

        if result.get("error"):
            return {"task": task_description, "status": status, "error": result["error"]}
        return result.get("todo", {"task": task_description, "status": status})

    def _delegate_search(task: str, details: str = "", _max_search_operations: int = max_search_operations) -> str:
        """
        Delegate a research task to the Search Agent.

        The Search Agent will execute up to 5 searches to answer your question,
        and write findings to a dedicated file. The summary is returned to you
        immediately, and the full findings will be available when you synthesize
        your final report.
        """
        tasks = _parse_tasks(task)
        if not tasks:
            raise ValueError("At least one todo description is required.")

        summaries = []

        for subtask in tasks:
            todo = _ensure_todo(subtask)
            _mark_status(todo["task"], "active")

            sequence_counter["count"] += 1
            sequence = sequence_counter["count"]

            try:
                result = run_search_agent(
                    task=subtask,
                    details=details or task,
                    llm=llm,
                    db_path=db_path,
                    findings_dir=findings_dir,
                    todos_path=todos_path,
                    embedding_model=embedding_model,
                    run_uuid=run_uuid,
                    sequence=sequence,
                    token_counter=token_counter,
                    activity_logger=logger,
                    display_callback=display_callback,
                    max_search_operations=_max_search_operations,
                )
            except Exception:
                _mark_status(todo["task"], "pending")
                raise

            summary = result.get("summary", "Search Agent completed but returned no summary.")
            findings_file = result.get("findings_file", "")

            _mark_status(todo["task"], "complete")

            bullet = f"- **{subtask}**: {summary}"
            if findings_file:
                bullet += f" (📄 {findings_file})"
            summaries.append(bullet)

        header = f"Delegated tasks under '{task}':"
        body = "\n".join(summaries)
        return f"{header}\n{body}"

    return StructuredTool.from_function(
        func=_delegate_search,
        name="delegate_search",
        description=(
            f"Delegate 1-3 research subtasks to the Search Agent (each gets up to {max_search_operations} searches). "
            "Automatically creates and manages the related todos."
        ),
        args_schema=DelegateSearchInput,
    )
