"""Shared helpers for commands that depend on the ``uv`` CLI."""

import functools
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from vision_agents.cli.errors import CliError

F = TypeVar("F", bound=Callable[..., object])


def ensure_uv() -> None:
    """Raise ``CliError`` if ``uv`` is not on ``PATH``."""
    if shutil.which("uv") is None:
        raise CliError("`uv` is required. Install it from https://docs.astral.sh/uv/.")


def requires_uv(func: F) -> F:
    """Decorator that asserts ``uv`` is available before invoking ``func``."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        ensure_uv()
        return func(*args, **kwargs)

    return cast(F, wrapper)


@requires_uv
def install_dependencies(target: Path) -> None:
    """Run ``uv sync`` in ``target`` to provision its venv.

    Stdout/stderr are captured: on success they are dropped (the install
    succeeded, the user does not need 80 lines of "+ package==x.y.z"),
    on failure they are surfaced in the error message so the user can
    diagnose what broke.
    """
    try:
        subprocess.run(
            ["uv", "sync"],
            cwd=target,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        output = (err.stderr or err.stdout or "").strip()
        detail = f"\n{output}" if output else ""
        raise CliError(
            f"'uv sync' failed with exit code {err.returncode}{detail}"
        ) from err
