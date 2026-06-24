"""Resolve the agent entrypoint from settings or the ``--entrypoint`` flag."""

from pathlib import Path

from vision_agents.cli.agent.models import ResolvedEntrypoint
from vision_agents.cli.errors import CliError
from vision_agents.config import (
    TOOL_TABLE,
    AgentSettings,
    Settings,
    load_settings,
    parse_entrypoint,
)

CONFIG_KEY = f"{TOOL_TABLE}.agent.entrypoint"


def resolve_entrypoint(cwd: Path, override: str | None) -> ResolvedEntrypoint:
    """Resolve the agent entrypoint from ``--entrypoint`` or ``pyproject.toml``.

    The override takes precedence and lets the user run without a config file.
    """
    if override is not None:
        try:
            module, attribute = parse_entrypoint(override)
        except ValueError as err:
            raise CliError(f"--entrypoint: {err}") from err
        return ResolvedEntrypoint(
            project_root=cwd.resolve(),
            module=module,
            attribute=attribute,
            config_path=None,
        )

    settings = _load_or_raise(cwd)
    agent = _require_agent(settings)
    return ResolvedEntrypoint(
        project_root=settings.project_root,
        module=agent.module,
        attribute=agent.attribute,
        config_path=settings.pyproject_path,
    )


def _load_or_raise(cwd: Path) -> Settings:
    try:
        return load_settings(cwd)
    except FileNotFoundError as err:
        raise CliError(
            "Could not find pyproject.toml; you may not be in a Vision Agents project."
        ) from err
    except ValueError as err:
        raise CliError(str(err)) from err


def _require_agent(settings: Settings) -> AgentSettings:
    if settings.agent is None:
        raise CliError(
            f"{settings.pyproject_path} is missing required "
            f"[{TOOL_TABLE}.agent] section"
        )
    return settings.agent
