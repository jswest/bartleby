from rich.console import Console
from rich.text import Text

SPLASH = """
 ██████╗  █████╗ ██████╗ ████████╗██╗     ███████╗██████╗ ██╗   ██╗
 ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║     ██╔════╝██╔══██╗╚██╗ ██╔╝
 ██████╔╝███████║██████╔╝   ██║   ██║     █████╗  ██████╔╝ ╚████╔╝
 ██╔══██╗██╔══██║██╔══██╗   ██║   ██║     ██╔══╝  ██╔══██╗  ╚██╔╝
 ██████╔╝██║  ██║██║  ██║   ██║   ███████╗███████╗██████╔╝   ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝╚═════╝    ╚═╝
An AI-powered scrivener who would definitely prefer not to.
"""

_console = Console(highlight=False)


def send(message: str | None = None, message_type: str | None = None):
    if message_type == "SPLASH":
        _console.print(SPLASH, style="magenta")
    elif message_type == "BIG":
        _console.print("  ", message, style="bold cyan")
    elif message_type == "WARN":
        _console.print("  ", message, style="dim")
    elif message_type == "ERROR":
        _console.print("  ", message, style="bold red")
    elif message_type == "COMPLETE":
        _console.print("  ", message)
    elif message_type == "INFO":
        _console.print(message, style="dim")
    else:
        _console.print(message)


def print_session_info(project: str = "", model: str = "", doc_count: int = 0, session_id: str = ""):
    """Print a compact session info line below the splash."""
    parts = []
    if project:
        parts.append(f"Project: [bold]{project}[/bold]")
    if model:
        parts.append(f"Model: [bold]{model}[/bold]")
    if doc_count > 0:
        parts.append(f"Documents: [bold]{doc_count}[/bold]")
    if session_id:
        parts.append(f"Session: [dim]{session_id}[/dim]")

    if parts:
        _console.print("  " + " | ".join(parts))
        _console.print()
