"""Click error type with a colorized ``Error:`` prefix."""

import sys
from typing import IO

import click


class CliError(click.ClickException):
    """``ClickException`` that prints with a bold red ``Error:`` prefix in TTYs.

    Respects the color preference cached by ``ClickException`` at construction
    time via ``resolve_color_default()`` (which honors ``NO_COLOR``, the active
    click context's ``color`` setting, and tty detection).
    """

    def show(self, file: IO[str] | None = None) -> None:
        if file is None:
            file = sys.stderr
        prefix = click.style("Error:", fg="red", bold=True)
        click.echo(
            f"{prefix} {self.format_message()}",
            file=file,
            color=self.show_color,
        )
