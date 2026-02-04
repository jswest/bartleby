from rich.console import Console

SPLASH = """
 ███████████                       █████    ████           █████
░░███░░░░░███                     ░░███    ░░███          ░░███
 ░███    ░███  ██████   ████████  ███████   ░███   ██████  ░███████  █████ ████
 ░██████████  ░░░░░███ ░░███░░███░░░███░    ░███  ███░░███ ░███░░███░░███ ░███
 ░███░░░░░███  ███████  ░███ ░░░   ░███     ░███ ░███████  ░███ ░███ ░███ ░███
 ░███    ░███ ███░░███  ░███       ░███ ███ ░███ ░███░░░   ░███ ░███ ░███ ░███
 ███████████ ░░████████ █████      ░░█████  █████░░██████  ████████  ░░███████
░░░░░░░░░░░   ░░░░░░░░ ░░░░░        ░░░░░  ░░░░░  ░░░░░░  ░░░░░░░░    ░░░░░███
                                                                      ███ ░███
                                                                     ░░██████
                                                                      ░░░░░░
An AI-powered scrivener who would definitely prefer not to.
-------------------------------------------------------------------------------
"""

_console = Console(highlight=False)


def send(message: str | None = None, message_type: str | None = None):
    if message_type == "SPLASH":
        _console.print(SPLASH, style="magenta")
    elif message_type == "BIG":
        _console.print("✎", message, style="bold cyan")
    elif message_type == "WARN":
        _console.print("⚠", message, style="dim")
    elif message_type == "ERROR":
        _console.print("☠", message, style="bold red")
    elif message_type == "COMPLETE":
        _console.print("☑", message)
    else:
        _console.print(message)
