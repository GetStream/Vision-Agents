"""Shared helpers for commands that depend on the ``uv`` CLI."""

import functools
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import click

F = TypeVar("F", bound=Callable[..., object])


def ensure_uv() -> None:
    """Raise ``ClickException`` if ``uv`` is not on ``PATH``."""
    if shutil.which("uv") is None:
        raise click.ClickException(
            "`uv` is required. Install it from https://docs.astral.sh/uv/."
        )


def requires_uv(func: F) -> F:
    """Decorator that asserts ``uv`` is available before invoking ``func``."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        ensure_uv()
        return func(*args, **kwargs)

    return cast(F, wrapper)


@requires_uv
def install_dependencies(target: Path) -> None:
    """Run ``uv sync`` in ``target`` to provision its venv."""
    try:
        subprocess.run(["uv", "sync"], cwd=target, check=True)
    except subprocess.CalledProcessError as err:
        raise click.ClickException(
            f"'uv sync' failed with exit code {err.returncode}"
        ) from err
