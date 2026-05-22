import sys

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

# Status messages go to stderr so they don't collide with JSON output on
# stdout from skill scripts, and so they share a Console instance with the
# scribe progress bar — Rich's Live display can then insert our prints
# above the bar instead of stomping it.
_console = Console(highlight=False, file=sys.stderr)


def get_console() -> Console:
    """The shared Rich console for all status output.

    Pass to ``Progress(console=...)`` etc. so Live displays coordinate
    with our prints.
    """
    return _console


# Continuation-line indent for multi-line messages. Matches the column at which
# content lands after Rich's print("  ", message) prefix + separator (3 spaces).
_CONT_INDENT = "   "


def _aligned(message: str) -> str:
    return message.replace("\n", "\n" + _CONT_INDENT)


def splash() -> None:
    _console.print(SPLASH, style="magenta")


def big(message: str) -> None:
    _console.print("  ", _aligned(message), style="bold yellow")


def warn(message: str) -> None:
    _console.print("  ", _aligned(message), style="dim")


def error(message: str) -> None:
    _console.print("  ", _aligned(message), style="bold red")


def complete(message: str) -> None:
    _console.print("  ", _aligned(message))


def info(message: str) -> None:
    _console.print(message, style="dim")
