from rich.console import Console

SPLASH = """
 ██████╗  █████╗ ██████╗ ████████╗██╗     ███████╗██████╗ ██╗   ██╗
 ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║     ██╔════╝██╔══██╗╚██╗ ██╔╝
 ██████╔╝███████║██████╔╝   ██║   ██║     █████╗  ██████╔╝  ╚████╔╝
 ██╔══██╗██╔══██║██╔══██╗   ██║   ██║     ██╔══╝  ██╔══██╗   ╚██╔╝
 ██████╔╝██║  ██║██║  ██║   ██║   ███████╗███████╗██████╔╝    ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝╚═════╝     ╚═╝
An AI-powered scrivener who would definitely prefer not to.
"""

_console = Console(highlight=False)


def splash() -> None:
    _console.print(SPLASH, style="magenta")


def big(message: str) -> None:
    _console.print("  ", message, style="bold yellow")


def warn(message: str) -> None:
    _console.print("  ", message, style="dim")


def error(message: str) -> None:
    _console.print("  ", message, style="bold red")


def complete(message: str) -> None:
    _console.print("  ", message)


def info(message: str) -> None:
    _console.print(message, style="dim")
