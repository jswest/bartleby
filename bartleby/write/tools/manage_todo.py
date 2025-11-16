"""Tool for managing the agent's todo list."""

from typing import Dict, Any, Optional, Callable

from langchain_core.tools import tool

from bartleby.write.memory import TodoList


def create_manage_todo_tool(
    todo_list: TodoList,
    before_hook: Optional[Callable[[str], Any]] = None,
):
    """
    Create todo management tool.

    Args:
        todo_list: TodoList instance for storing todos
        before_hook: Optional hook called before tool execution

    Returns:
        LangChain tool instance
    """

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
        if before_hook:
            preempt = before_hook("manage_todo_tool")
            if preempt is not None:
                return preempt

        action_normalized = (action or "").strip().lower()

        if action_normalized == "add":
            if not task.strip():
                return {"error": "Task description is required when action='add'."}
            result = todo_list.add_todo(task)
            return {
                "message": result.get("message"),
                "todo": result.get("todo"),
                "total_todos": result.get("total_todos")
            }

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

    return manage_todo_tool
