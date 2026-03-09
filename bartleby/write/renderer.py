"""CLI renderer — subscribes to EventBus, handles all display output."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.prompt import Confirm
from smolagents import LogLevel
from smolagents.monitoring import AgentLogger

from bartleby.lib.console import SPLASH
from bartleby.write.events import (
    AnswerEvent,
    BudgetWarningEvent,
    ErrorEvent,
    Event,
    EventBus,
    ReadyEvent,
    ResearchSummaryEvent,
    SessionInfoEvent,
    SplashEvent,
    StatusEvent,
    ThinkingEvent,
    TokenStatsEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from bartleby.write.markdown import render_markdown
from bartleby.write.progress import ProgressDisplay


class CliRenderer:
    """Renders events to the terminal via Rich."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.progress: ProgressDisplay | None = None
        self._live: Live | None = None

    def on_event(self, event: Event) -> None:
        """Dispatch event to the appropriate handler."""
        handler = {
            SplashEvent: self._on_splash,
            StatusEvent: self._on_status,
            SessionInfoEvent: self._on_session_info,
            ReadyEvent: self._on_ready,
            ToolStartEvent: self._on_tool_start,
            ToolCompleteEvent: self._on_tool_complete,
            ThinkingEvent: self._on_thinking,
            ErrorEvent: self._on_error,
            AnswerEvent: self._on_answer,
            ResearchSummaryEvent: self._on_research_summary,
            TokenStatsEvent: self._on_token_stats,
            BudgetWarningEvent: self._on_budget_warning,
        }.get(type(event))
        if handler:
            handler(event)

    # -- Lifecycle --

    def start_live(self, max_steps: int, token_counter=None) -> None:
        self.progress = ProgressDisplay(max_steps=max_steps, token_counter=token_counter)
        self._live = Live(self.progress, console=self.console, refresh_per_second=4)
        self._live.start()

    def stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        self.progress = None

    def pause_live(self) -> None:
        if self._live:
            self._live.stop()

    def resume_live(self) -> None:
        if self._live:
            self._live.start()

    def prompt_step_extension(self, reason: str, requested: int) -> bool:
        self.console.print("")
        self.console.print(
            f"[bold yellow]Agent requests {requested} more steps:[/bold yellow] {reason}"
        )
        return Confirm.ask("Allow?", default=False, console=self.console)

    # -- Handlers --

    def _on_splash(self, event: SplashEvent) -> None:
        self.console.print(SPLASH, style="magenta")

    def _on_status(self, event: StatusEvent) -> None:
        self.console.print(f"  [bold yellow]{event.message}[/bold yellow]")

    def _on_session_info(self, event: SessionInfoEvent) -> None:
        parts = []
        if event.project:
            parts.append(f"Project: [bold]{event.project}[/bold]")
        if event.model:
            parts.append(f"Model: [bold]{event.model}[/bold]")
        if event.doc_count > 0:
            parts.append(f"Documents: [bold]{event.doc_count}[/bold]")
        if event.session_id:
            parts.append(f"Session: [dim]{event.session_id}[/dim]")
        if parts:
            self.console.print("  " + " | ".join(parts))
            self.console.print()

    def _on_ready(self, event: ReadyEvent) -> None:
        self.console.print(
            "[bold]Ask questions about your documents. "
            "Type /save to save the last answer as a report.[/bold]"
        )
        self.console.print(
            "[bold]Type /browse <#> to view a cited source passage in context.[/bold]"
        )
        self.console.print("[dim]Press Ctrl+C to exit.[/dim]\n")

    def _on_tool_start(self, event: ToolStartEvent) -> None:
        if self.progress:
            if event.is_subagent:
                self.progress.start_subtool(event.tool_name, event.tool_args)
            else:
                self.progress.start_tool(event.tool_name, event.tool_args)

    def _on_tool_complete(self, event: ToolCompleteEvent) -> None:
        if self.progress:
            if event.is_subagent:
                self.progress.complete_subtool(event.tool_name, event.observations, event.elapsed)
            else:
                self.progress.complete_tool(event.tool_name, event.observations)

    def _on_thinking(self, event: ThinkingEvent) -> None:
        if self.progress:
            self.progress.start_thinking()

    def _on_error(self, event: ErrorEvent) -> None:
        if self.progress:
            self.progress.add_error_line(event.message)
        else:
            self.console.print(f"[red]{event.message}[/red]")

    def _on_answer(self, event: AnswerEvent) -> None:
        self.console.print("")
        render_markdown(self.console, event.text)
        self.console.print("")

    def _on_research_summary(self, event: ResearchSummaryEvent) -> None:
        self.console.print("")
        self.console.print("[dim]" + "\u2500" * 60 + "[/dim]")
        self.console.print(
            f"[dim italic]Research: {event.summary.strip()}[/dim italic]"
        )
        self.console.print("[dim]" + "\u2500" * 60 + "[/dim]")

    def _on_token_stats(self, event: TokenStatsEvent) -> None:
        self.console.print(f"[dim]{event.stats}[/dim]\n")

    def _on_budget_warning(self, event: BudgetWarningEvent) -> None:
        self.console.print(f"[yellow]{event.message}[/yellow]")


class EventCapturingLogger(AgentLogger):
    """Captures smolagents log output and emits events instead of printing."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(level=LogLevel.ERROR, console=Console(quiet=True))
        self.event_bus = event_bus

    def log(self, *args, level=LogLevel.INFO, **kwargs) -> None:
        pass

    def log_error(self, error_message: str) -> None:
        msg = str(error_message)
        # Parsing errors are retryable — smolagents feeds the error back to the
        # model on the next step. Don't show them as errors.
        if "Error while parsing tool call" in msg:
            return
        if "Reached max steps" in msg:
            return
        if "Error while generating output" in msg:
            return
        if "Error executing request to team member" in msg:
            return
        self.event_bus.emit(ErrorEvent(message=msg))
