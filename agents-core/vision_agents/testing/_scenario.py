"""Scenario definition and YAML loader for multi-turn simulations."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ScenarioValue = str | bool | int | float

_REQUIRED_FIELDS = ("name", "goal", "success")


@dataclass(frozen=True)
class Scenario:
    """A simulated-user brief plus the criteria a conversation must satisfy.

    Attributes:
        name: Identifier used in reports.
        goal: What the simulated user is trying to achieve.
        success: Criteria the judge evaluates over the full transcript.
        mode: Conversation mode. Only ``"text"`` is supported.
        persona: Behavioural traits of the simulated user.
        context: Facts the simulated user knows and must not alter.
        constraints: Rules the simulated user follows while conversing.
        variations: Number of conversations to hold with reworded briefs.
            The first always uses the scenario as written.
        repeat: Number of times each variation is run for pass@k / pass^k.
        judge_target: Model name the scenario was written to be judged by.
    """

    name: str
    goal: str
    success: list[str]
    mode: str = "text"
    persona: dict[str, ScenarioValue] = field(default_factory=dict)
    context: dict[str, ScenarioValue] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    variations: int = 1
    repeat: int = 1
    judge_target: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.name, "name")
        _require_str(self.goal, "goal")
        if self.mode != "text":
            raise ValueError(f"Scenario field 'mode' must be 'text', got {self.mode!r}")
        _require_str_list(self.success, "success")
        if not self.success:
            raise ValueError(
                "Scenario field 'success' must list at least one criterion"
            )
        _require_str_list(self.constraints, "constraints")
        _require_mapping(self.persona, "persona")
        _require_mapping(self.context, "context")
        _require_positive_int(self.variations, "variations")
        _require_positive_int(self.repeat, "repeat")
        if self.judge_target is not None:
            _require_str(self.judge_target, "judge_target")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        """Build a scenario from a parsed YAML mapping.

        Raises:
            ValueError: If a required field is missing or an unknown field is present.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown scenario field(s): {', '.join(repr(k) for k in unknown)}"
            )
        missing = [name for name in _REQUIRED_FIELDS if name not in data]
        if missing:
            raise ValueError(
                f"Scenario is missing required field(s): {', '.join(repr(k) for k in missing)}"
            )
        return cls(**data)

    @property
    def brief(self) -> str:
        """Human-readable version of the scenario handed to the simulated user."""
        lines = [f"Goal: {self.goal}"]
        if self.persona:
            lines.append("Persona:")
            lines.extend(f"- {k}: {v}" for k, v in self.persona.items())
        if self.context:
            lines.append(
                "Facts about you (use exactly as given, never invent or change them):"
            )
            lines.extend(f"- {k}: {v}" for k, v in self.context.items())
        if self.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {c}" for c in self.constraints)
        return "\n".join(lines)


def load_scenario(path: str | Path) -> Scenario:
    """Load a single scenario from a YAML file.

    Raises:
        ValueError: If the file is not a mapping or fails scenario validation.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file {path} must contain a YAML mapping")
    return Scenario.from_dict(data)


def _require_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Scenario field {name!r} must be a non-empty string")


def _require_str_list(value: object, name: str) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"Scenario field {name!r} must be a list of strings")


def _require_mapping(value: object, name: str) -> None:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"Scenario field {name!r} must be a mapping with string keys")
    for key, item in value.items():
        if not isinstance(item, (str, bool, int, float)):
            raise ValueError(
                f"Scenario field {name!r} entry {key!r} must be a string, number or boolean"
            )


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Scenario field {name!r} must be a positive integer")
