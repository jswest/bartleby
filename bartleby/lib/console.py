from rich.console import Console
from rich.live import Live

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

def send(message: str | None = None, message_type: str | None = None):
    console = Console(highlight=False)
    if message_type == "SPLASH":
        console.print(SPLASH, style="magenta")
    elif message_type == "BIG":
        console.print("✎", message, style="bold cyan")
    elif message_type == "WARN":
        console.print("⚠", message, style="dim")
    elif message_type == "ERROR":
        console.print("☠", message, style="bold red")
    elif message_type == "COMPLETE":
        console.print("☑", message)
    elif message_type == "INCOMPLETE":
        console.print("☐", message)
    elif message_type == "ACTION":
        console.print("•", message)
    elif message_type == "ACTION_DIM":
        console.print("•", message, style="dim")
    elif message_type == "ACTION_TODO_ACTIVE":
        console.print("☐", message, style="bold")
    elif message_type == "ACTION_TODO_COMPLETE":
        console.print("☑", message)
    elif message_type == "ACTION_TODO_PENDING":
        console.print("☐", message)
    elif message_type == "TOKENS":
        console.print(f"[Tokens: {message}]", style="dim italic")
    elif message_type == "TODOS":
        console.print(message)
    elif message_type == "REPORT":
        from rich.markdown import Markdown
        console.print(Markdown(message))
    else:
        console.print(message)
    