"""Shared helpers for locating and parsing ``vision-agents.toml``."""

import tomllib
from pathlib import Path

import click

CONFIG_FILENAME = "vision-agents.toml"


def find_config(start: Path) -> Path:
    """Walk up from ``start`` looking for ``vision-agents.toml``.

    Raises:
        click.ClickException: if no config file is found.
    """
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise click.ClickException(
        "Could not find vision-agents configuration; "
        "you may not be in a Vision Agents project."
    )


def load_app_config(config_path: Path) -> dict[str, object]:
    """Parse the ``[app]`` section of ``vision-agents.toml``.

    Strict on required fields and types; unknown keys are ignored for
    forward compatibility.
    """
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as err:
        raise click.ClickException(f"failed to read {config_path}: {err}") from err
    except tomllib.TOMLDecodeError as err:
        raise click.ClickException(f"failed to parse {config_path}: {err}") from err

    app = data.get("app")
    if not isinstance(app, dict):
        raise click.ClickException(f"{config_path} is missing required [app] section")

    entrypoint = app.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise click.ClickException(
            f"{config_path} requires app.entrypoint to be a non-empty string"
        )

    return {"entrypoint": entrypoint}
