"""Memory utilities for agent scratchpad and todo list."""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any


def read_scratchpad(scratchpad_path: Path) -> str:
    """
    Read the full contents of the scratchpad.

    Args:
        scratchpad_path: Path to the scratchpad file

    Returns:
        Contents of scratchpad.md or empty string if doesn't exist
    """

    if not scratchpad_path.exists():
        return ""

    return scratchpad_path.read_text(encoding="utf-8")


def append_to_scratchpad(scratchpad_path: Path, content: str) -> str:
    """
    Append content to the scratchpad with a timestamp.

    Args:
        scratchpad_path: Path to the scratchpad file
        content: Content to append

    Returns:
        Success message
    """
    timestamp = datetime.now().isoformat()
    entry = f"\n\n---\n**{timestamp}**\n\n{content}\n"

    with scratchpad_path.open("a", encoding="utf-8") as f:
        f.write(entry)

    return f"Appended to scratchpad at {timestamp}"


class TodoList:
    """In-memory todo list for agent task tracking."""

    def __init__(self, todos_path: str):
        self.todos_path = todos_path
        self.todos: List[Dict[str, str]] = []

    def write_todo_list(self):
        """Write the current todos to the JSON file."""
        with open(self.todos_path, 'w', encoding='utf-8') as out_file:
            json.dump(self.todos, out_file, indent=2)

    def add_todo(self, task: str) -> Dict[str, Any]:
        """
        Add a new todo item.

        Args:
            task: Task description

        Returns:
            Dictionary with the new todo and current list
        """
        todo = {"task": task, "status": "pending"}
        self.todos.append(todo)
        self.write_todo_list()
        return {
            "message": f"Added todo: {task}",
            "todo": todo,
            "total_todos": len(self.todos),
        }

    def update_todo_status(self, task: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a todo item.

        Args:
            task: Task description to match (case-insensitive substring match)
            status: New status (pending, active, or complete)

        Returns:
            Dictionary with update result
        """
        if status not in ["pending", "active", "complete"]:
            return {"error": f"Invalid status: {status}. Must be pending, active, or complete"}

        # Find matching todo (case-insensitive substring match)
        task_lower = task.lower()
        matching_todos = [
            (i, todo) for i, todo in enumerate(self.todos)
            if task_lower in todo["task"].lower()
        ]

        if not matching_todos:
            return {"error": f"No todo found matching: {task}"}

        if len(matching_todos) > 1:
            return {
                "error": f"Multiple todos match '{task}'. Please be more specific.",
                "matches": [todo["task"] for _, todo in matching_todos],
            }

        # Update the todo
        idx, todo = matching_todos[0]
        old_status = todo["status"]
        self.todos[idx]["status"] = status

        self.write_todo_list()
        return {
            "message": f"Updated '{todo['task']}' from {old_status} to {status}",
            "todo": self.todos[idx],
        }

    def get_todos(self) -> Dict[str, Any]:
        """
        Get all todos grouped by status.

        Returns:
            Dictionary with todos organized by status
        """
        pending = [t for t in self.todos if t["status"] == "pending"]
        active = [t for t in self.todos if t["status"] == "active"]
        complete = [t for t in self.todos if t["status"] == "complete"]

        self.write_todo_list()
        return {
            "total": len(self.todos),
            "pending": pending,
            "active": active,
            "complete": complete,
        }

    def clear_todos(self) -> Dict[str, str]:
        """
        Clear all todos.

        Returns:
            Confirmation message
        """
        count = len(self.todos)
        self.todos = []
        self.write_todo_list()
        return {"message": f"Cleared {count} todos"}

    def get_all_todos(self) -> List[Dict[str, str]]:
        """
        Return a shallow copy of all todos.
        """
        return list(self.todos)
