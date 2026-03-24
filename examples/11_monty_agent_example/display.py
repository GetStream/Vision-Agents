"""Pretty terminal output for demo purposes."""

import contextlib
import logging
from collections.abc import Generator

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.syntax import Syntax

    _console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


@contextlib.contextmanager
def thinking_spinner(message: str = "Crafting code") -> Generator[None, None, None]:
    """Show an animated spinner while the agent is working."""
    if _HAS_RICH:
        with Live(
            Spinner("dots", text=f"[bold yellow] {message}...[/bold yellow]"),
            console=_console,
            transient=True,
        ):
            yield
    else:
        print(f"  ⏳ {message}...")
        yield


def log_code(code: str) -> None:
    """Pretty-print Python code with syntax highlighting."""
    if _HAS_RICH:
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        _console.print(
            Panel(syntax, title="[bold cyan]Agent Code[/bold cyan]", border_style="cyan")
        )
    else:
        logger.warning(f"Executing code:\n{code}")


def log_result(result: str) -> None:
    """Pretty-print code execution result."""
    if _HAS_RICH:
        _console.print(
            Panel(result, title="[bold green]Result[/bold green]", border_style="green")
        )
    else:
        logger.info(f"Code result: {result}")


def log_transcript(role: str, text: str) -> None:
    """Print a transcript line."""
    emoji = "🗣️ " if role == "User" else "🤖"
    if _HAS_RICH:
        _console.print(f"  {emoji} [bold]{role}:[/bold] {text}")
    else:
        print(f"  {emoji} {role}: {text}")


class TranscriptAccumulator:
    """Collects transcript deltas and flushes on sentence boundaries."""

    def __init__(self, role: str) -> None:
        self._role = role
        self._parts: list[str] = []

    def push(self, text: str, mode: str) -> None:
        self._parts.append(text)
        if mode == "final" or text.endswith((".", "?", "!")):
            full = "".join(self._parts).strip()
            if full:
                log_transcript(self._role, full)
            self._parts.clear()
