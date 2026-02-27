from dataclasses import dataclass, field, fields
from typing import Any, Self


@dataclass
class SessionInfo:
    """Represents a session registered in the session registry."""

    session_id: str
    call_id: str
    node_id: str
    started_at: float
    metrics_updated_at: float
    metrics: dict[str, int | float | None] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct from a dict, silently ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
