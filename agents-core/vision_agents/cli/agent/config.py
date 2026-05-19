"""Locate and parse the project's ``pyproject.toml`` agent config."""

import tomllib
from pathlib import Path

from vision_agents.cli.agent.models import ResolvedEntrypoint
from vision_agents.cli.errors import CliError

CONFIG_FILENAME = "pyproject.toml"
CONFIG_KEY = "tool.vision-agents.agent.entrypoint"


def config_hint(config_path: Path, spec: str) -> str:
    """User-facing hint pointing at the config key that owns ``spec``."""
    return f'Check {CONFIG_KEY} = "{spec}" in {config_path}.'


def parse_entrypoint(spec: object) -> tuple[str, str]:
    """Parse a gunicorn-style ``module:attribute`` spec.

    Raises:
        ValueError: if ``spec`` is not a non-empty ``module:attribute`` string.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"entrypoint {spec!r} must be in 'module:attribute' form (e.g. 'agent:runner')"
        )
    module, _, attribute = spec.partition(":")
    if not module or not attribute:
        raise ValueError(
            f"entrypoint {spec!r} is malformed; expected 'module:attribute'"
        )
    return module, attribute


def find_config(start: Path) -> Path:
    """Walk up from ``start`` looking for ``pyproject.toml``.

    Raises:
        CliError: if no ``pyproject.toml`` is found.
    """
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise CliError(
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
        raise CliError(f"failed to read {config_path}: {err}") from err
    except tomllib.TOMLDecodeError as err:
        raise CliError(f"failed to parse {config_path}: {err}") from err

    tool = data.get("tool")
    va = tool.get("vision-agents") if isinstance(tool, dict) else None
    section = va.get("agent") if isinstance(va, dict) else None
    if not isinstance(section, dict):
        raise CliError(
            f"{config_path} is missing required [tool.vision-agents.agent] section"
        )

    try:
        module, attribute = parse_entrypoint(section.get("entrypoint"))
    except ValueError as err:
        raise CliError(
            f"{config_path} requires tool.vision-agents.agent.entrypoint "
            "in 'module:attribute' form (e.g. 'agent:runner')"
        ) from err

    return {"module": module, "attribute": attribute}


def resolve_entrypoint(cwd: Path, override: str | None) -> ResolvedEntrypoint:
    """Resolve the agent entrypoint from ``--entrypoint`` or ``pyproject.toml``.

    The override takes precedence and lets the user run without a config file.
    """
    if override is not None:
        try:
            module, attribute = parse_entrypoint(override)
        except ValueError as err:
            raise CliError(
                f"--entrypoint must be in 'module:attribute' form (got {override!r})"
            ) from err
        return ResolvedEntrypoint(
            project_root=cwd.resolve(),
            module=module,
            attribute=attribute,
            hint=f"Check --entrypoint={override!r}.",
        )

    config_path = find_config(cwd)
    config = load_agent_config(config_path)
    module = str(config["module"])
    attribute = str(config["attribute"])
    return ResolvedEntrypoint(
        project_root=config_path.parent.resolve(),
        module=module,
        attribute=attribute,
        hint=config_hint(config_path, f"{module}:{attribute}"),
    )
