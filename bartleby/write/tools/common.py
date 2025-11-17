"""Shared utilities for bartleby tools."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from functools import wraps

from bartleby.lib.consts import DEFAULT_SEARCH_RESULT_LIMIT, MAX_SEARCH_RESULT_LIMIT
from bartleby.read.sqlite import get_connection


def sanitize_limit(limit: Optional[int]) -> int:
    """Clamp search limits to keep tool outputs lean."""
    if limit is None:
        return DEFAULT_SEARCH_RESULT_LIMIT
    return max(1, min(limit, MAX_SEARCH_RESULT_LIMIT))


def result_metadata(results: List) -> List[Dict[str, Any]]:
    """Convert search results to metadata dictionaries."""
    return [r.to_metadata_dict() for r in results]


def document_exists(db_path: Path, document_id: str) -> bool:
    """Return True if the provided document_id exists in the documents table."""
    connection = get_connection(db_path)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
        (document_id,),
    )
    exists = cursor.fetchone() is not None
    connection.close()
    return exists


def with_hook(tool_name: str, before_hook: Optional[Callable] = None):
    """
    Decorator that executes before_hook and allows preemptive return.

    Args:
        tool_name: Name of the tool (for hook identification)
        before_hook: Optional hook function that may return a preemptive response

    Returns:
        Decorated function that runs the hook before execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if before_hook:
                preempt = before_hook(tool_name)
                if preempt is not None:
                    return preempt
            return func(*args, **kwargs)
        return wrapper
    return decorator
