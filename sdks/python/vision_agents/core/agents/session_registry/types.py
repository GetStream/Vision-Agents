from dataclasses import dataclass, field, fields
from typing import Any, Self

from vision_agents.core.observability.agent import AgentMetrics


@dataclass
class SessionInfo:
    """Represents a session registered in the session registry."""

    session_id: str
    call_id: str
    node_id: str
    started_at: float
    metrics_updated_at: float
    metrics: AgentMetrics = field(default_factory=AgentMetrics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "session_id": self.session_id,
            "call_id": self.call_id,
            "node_id": self.node_id,
            "started_at": self.started_at,
            "metrics_updated_at": self.metrics_updated_at,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct from a dict, silently ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        metrics_raw = filtered.get("metrics")
        if metrics_raw is not None:
            filtered["metrics"] = AgentMetrics.from_dict(metrics_raw)
        return cls(**filtered)
