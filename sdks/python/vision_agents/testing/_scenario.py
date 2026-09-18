"""Scenario files for ``vision-agents agent simulate``.

A scenario is a TOML file with a brief for the simulated caller and the
criteria the agent has to meet::

    name = "refund"
    scenario = "You bought shoes last week that do not fit and want a refund."
    criteria = [
      "The agent asks for the order number",
      "The agent explains the refund policy",
    ]
    max_turns = 8
    variations = 1
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_TURNS = 12
SCENARIO_SUFFIX = ".toml"


class ScenarioError(ValueError):
    """A scenario file is missing, unreadable or malformed."""


@dataclass
class Scenario:
    """A brief for the simulated caller plus the criteria the agent must meet.

    Attributes:
        name: Display name; defaults to the file stem.
        scenario: What the caller wants, in prose. A brief, not a script.
        criteria: Statements the judge checks against the finished transcript.
        max_turns: How many times the caller may speak before the conversation is cut off.
        variations: How many ways of phrasing the brief to try; the brief as written is always the first.
        path: Source file, when loaded from disk.
    """

    name: str
    scenario: str
    criteria: list[str]
    max_turns: int = DEFAULT_MAX_TURNS
    variations: int = 1
    path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ScenarioError("name must not be empty")
        if not self.scenario.strip():
            raise ScenarioError("scenario must not be empty")
        if not self.criteria or not all(c.strip() for c in self.criteria):
            raise ScenarioError(
                "criteria must be a non-empty list of non-empty strings"
            )
        if self.max_turns < 1:
            raise ScenarioError("max_turns must be >= 1")
        if self.variations < 1:
            raise ScenarioError("variations must be >= 1")


def load_scenario(path: Path) -> Scenario:
    """Load and validate one scenario file.

    Raises:
        ScenarioError: if the file cannot be read or does not describe a scenario.
    """
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as err:
        raise ScenarioError(f"failed to read {path}: {err}") from err
    except tomllib.TOMLDecodeError as err:
        raise ScenarioError(f"failed to parse {path}: {err}") from err

    name = data.get("name", path.stem)
    scenario = data.get("scenario")
    criteria = data.get("criteria")
    max_turns = data.get("max_turns", DEFAULT_MAX_TURNS)
    variations = data.get("variations", 1)

    if not isinstance(name, str):
        raise ScenarioError(f"{path}: name must be a string")
    if not isinstance(scenario, str):
        raise ScenarioError(f"{path}: scenario is required and must be a string")
    if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
        raise ScenarioError(
            f"{path}: criteria is required and must be a list of strings"
        )
    if isinstance(max_turns, bool) or not isinstance(max_turns, int):
        raise ScenarioError(f"{path}: max_turns must be an integer")
    if isinstance(variations, bool) or not isinstance(variations, int):
        raise ScenarioError(f"{path}: variations must be an integer")

    try:
        return Scenario(
            name=name,
            scenario=scenario,
            criteria=criteria,
            max_turns=max_turns,
            variations=variations,
            path=path,
        )
    except ScenarioError as err:
        raise ScenarioError(f"{path}: {err}") from err


def find_scenarios(target: Path) -> list[Path]:
    """Return the scenario files at ``target``.

    A file is returned as-is; a directory yields every ``*.toml`` directly in it, sorted.

    Raises:
        ScenarioError: if ``target`` does not exist.
    """
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.glob(f"*{SCENARIO_SUFFIX}") if p.is_file())
    raise ScenarioError(f"{target} does not exist")
