"""Typed event architecture for the write command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    """Base class for all events."""


@dataclass
class SplashEvent(Event):
    pass


@dataclass
class StatusEvent(Event):
    message: str


@dataclass
class SessionInfoEvent(Event):
    project: str
    model: str
    doc_count: int
    session_id: str


@dataclass
class ReadyEvent(Event):
    pass


@dataclass
class ToolStartEvent(Event):
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    is_subagent: bool = False


@dataclass
class ToolCompleteEvent(Event):
    tool_name: str
    observations: str = ""
    elapsed: float = 0.0
    is_subagent: bool = False


@dataclass
class ThinkingEvent(Event):
    pass


@dataclass
class ErrorEvent(Event):
    message: str
    is_retryable: bool = True


@dataclass
class AnswerEvent(Event):
    text: str


@dataclass
class ResearchSummaryEvent(Event):
    summary: str


@dataclass
class TokenStatsEvent(Event):
    stats: str


@dataclass
class BudgetWarningEvent(Event):
    message: str


class EventBus:
    """Simple pub/sub event bus."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[Event], Any]] = []

    def subscribe(self, listener: Callable[[Event], Any]) -> None:
        self._listeners.append(listener)

    def emit(self, event: Event) -> None:
        for listener in self._listeners:
            listener(event)
