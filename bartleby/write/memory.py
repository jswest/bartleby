"""Memory utilities for agent todo list."""

import json
from typing import List, Dict, Any


class TodoList:
    """In-memory todo list for agent task tracking."""

    def __init__(self, todos_path: str):
        self.todos_path = todos_path
        self.todos: List[Dict[str, str]] = []
        self._load_from_file()

    def _load_from_file(self):
        """Load todos from the JSON file if it exists."""
        try:
            with open(self.todos_path, 'r', encoding='utf-8') as in_file:
                self.todos = json.load(in_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is invalid, start with empty list
            self.todos = []

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
        Reloads from file to ensure latest state.
        """
        self._load_from_file()
        return list(self.todos)
