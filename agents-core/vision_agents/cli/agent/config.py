"""Locate and parse the project's ``pyproject.toml`` agent config."""

import tomllib
from pathlib import Path

import click

CONFIG_FILENAME = "pyproject.toml"


def find_config(start: Path) -> Path:
    """Walk up from ``start`` looking for ``pyproject.toml``.

    Raises:
        click.ClickException: if no ``pyproject.toml`` is found.
    """
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise click.ClickException(
        "Could not find pyproject.toml; you may not be in a Vision Agents project."
    )


def load_agent_config(config_path: Path) -> dict[str, object]:
    """Parse ``[tool.vision-agents.agent]`` from ``pyproject.toml``.

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

    tool = data.get("tool")
    va = tool.get("vision-agents") if isinstance(tool, dict) else None
    section = va.get("agent") if isinstance(va, dict) else None
    if not isinstance(section, dict):
        raise click.ClickException(
            f"{config_path} is missing required [tool.vision-agents.agent] section"
        )

    entrypoint = section.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise click.ClickException(
            f"{config_path} requires tool.vision-agents.agent.entrypoint "
            "to be a non-empty string"
        )

    return {"entrypoint": entrypoint}
