"""Tool for delegating research tasks to the Search Agent."""

from pathlib import Path
from typing import Optional, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import tool

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

    def _mark_todo(task_description: str, status: str) -> str:
        """Update todo status if a matching todo exists."""
        if not todo_list:
            return ""
        try:
            result = todo_list.update_todo_status(task_description, status)
        except Exception:
            return ""
        if result.get("error"):
            return ""
        return result.get("todo", {}).get("task", task_description)

    @tool
    def delegate_search(task: str, details: str = "") -> str:
        """
        Delegate a research task to the Search Agent.

        The Search Agent will execute up to 5 searches to answer your question,
        and write findings to a dedicated file. The summary is returned to you
        immediately, and the full findings will be available when you synthesize
        your final report.

        Args:
            task: The research question or task description
            details: Additional context or specific things to look for

        Returns:
            Summary of what the Search Agent found
        """
        # Increment sequence
        sequence_counter["count"] += 1
        sequence = sequence_counter["count"]

        # Mark todo as active
        activated_todo = _mark_todo(task, "active")

        # Run the search agent
        result = run_search_agent(
            task=task,
            details=details,
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
        )

        # Mark todo as complete
        completed_todo = _mark_todo(task, "complete")

        # Build response
        summary = result.get("summary", "Search Agent completed but returned no summary.")
        findings_file = result.get("findings_file", "")

        follow_up_note = ""
        if activated_todo:
            follow_up_note += f"\n\n(🗂️ Marked todo '{activated_todo}' as active.)"
        if completed_todo:
            follow_up_note += f"\n\n(✅ Marked todo '{completed_todo}' complete.)"
        if findings_file:
            follow_up_note += f"\n\n(📄 Full findings saved to: {findings_file})"

        return summary + follow_up_note if follow_up_note else summary

    return delegate_search
