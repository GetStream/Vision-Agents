"""Project-level configuration for Vision Agents.

Reads ``[tool.vision-agents.*]`` from the nearest ``pyproject.toml``.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

PYPROJECT_FILENAME = "pyproject.toml"
TOOL_TABLE = "tool.vision-agents"


@dataclass(frozen=True)
class AgentSettings:
    """``[tool.vision-agents.agent]`` — how to invoke the project's ``Runner``."""

    module: str  # "agent" in "agent:runner"
    attribute: str  # "runner" in "agent:runner"

    @property
    def entrypoint(self) -> str:
        """Round-trip ``module:attribute`` representation."""
        return f"{self.module}:{self.attribute}"


@dataclass(frozen=True)
class Settings:
    """The resolved Vision Agents project configuration.

    Attributes:
        project_root: Directory containing the discovered ``pyproject.toml``.
        pyproject_path: Path to the discovered ``pyproject.toml`` itself.
        agent: Parsed ``[tool.vision-agents.agent]`` table or ``None``.
        raw: Full ``[tool.vision-agents]`` dict — escape hatch for plugin
            or user code reading custom subsections we don't model yet.
    """

    project_root: Path
    pyproject_path: Path
    agent: AgentSettings | None = None
    raw: dict[str, object] = field(default_factory=dict)


def find_pyproject(start: Path) -> Path:
    """Walk up from ``start`` looking for ``pyproject.toml``.

    Raises:
        FileNotFoundError: if no ``pyproject.toml`` is found anywhere on the
            way up. Callers should reformat to a domain-appropriate error.
    """
    for directory in (start, *start.parents):
        candidate = directory / PYPROJECT_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {PYPROJECT_FILENAME} found starting from {start}")


def parse_entrypoint(spec: object) -> tuple[str, str]:
    """Parse a gunicorn-style ``module:attribute`` spec.

    Raises:
        ValueError: if ``spec`` is not a non-empty ``module:attribute`` string.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"entrypoint {spec!r} must be in 'module:attribute' form "
            "(e.g. 'agent:runner')"
        )
    if spec.count(":") != 1:
        raise ValueError(
            f"entrypoint {spec!r} is malformed; expected exactly one ':' "
            "separator in 'module:attribute'"
        )
    module, _, attribute = spec.partition(":")
    if not module or not attribute:
        raise ValueError(
            f"entrypoint {spec!r} is malformed; expected 'module:attribute'"
        )
    if module.endswith(".py"):
        raise ValueError(
            f"entrypoint {spec!r} looks like a file path; entrypoint is a "
            f"Python import name, not a filename — try {module[:-3]!r}:{attribute!r} "
            f"(i.e. '{module[:-3]}:{attribute}')"
        )
    return module, attribute


def load_settings(cwd: Path | None = None) -> Settings:
    """Load Vision Agents settings from the nearest ``pyproject.toml``.

    Walks up from ``cwd`` (default: :func:`Path.cwd`) to find the project's
    ``pyproject.toml``, then parses ``[tool.vision-agents.*]`` sections into
    a :class:`Settings` instance.

    Args:
        cwd: Directory to start the upward search from. Defaults to the
            current working directory.

    Raises:
        FileNotFoundError: when no ``pyproject.toml`` is found.
        ValueError: when ``[tool.vision-agents.*]`` is malformed (wrong
            type for a known field, malformed entrypoint, etc.).
    """
    start = cwd if cwd is not None else Path.cwd()
    pyproject_path = find_pyproject(start)

    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)

    tool = data.get("tool")
    raw = tool.get("vision-agents") if isinstance(tool, dict) else None
    if not isinstance(raw, dict):
        raw = {}

    return Settings(
        project_root=pyproject_path.parent.resolve(),
        pyproject_path=pyproject_path,
        agent=_parse_agent(raw.get("agent")),
        raw=cast(dict[str, object], raw),
    )


def _parse_agent(section: object) -> AgentSettings | None:
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f"[{TOOL_TABLE}.agent] must be a table")
    module, attribute = parse_entrypoint(section.get("entrypoint"))
    return AgentSettings(module=module, attribute=attribute)
