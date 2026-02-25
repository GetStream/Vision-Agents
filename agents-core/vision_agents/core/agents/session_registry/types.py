from dataclasses import dataclass, field


@dataclass
class SessionInfo:
    """Represents a session registered in the session registry."""

    session_id: str
    call_id: str
    node_id: str
    started_at: float
    metrics_updated_at: float
    metrics: dict[str, int | float | None] = field(default_factory=dict)
