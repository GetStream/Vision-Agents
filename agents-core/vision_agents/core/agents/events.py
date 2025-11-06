from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent, BaseEvent
from typing import Optional, Any, Dict


@dataclass
class AgentInitEvent(BaseEvent):
    """Event emitted when Agent class initialized."""

    type: str = field(default="agent.init", init=False)


@dataclass
class AgentFinishEvent(BaseEvent):
    """Event emitted when agent.finish() call ended."""

    type: str = field(default="agent.finish", init=False)


@dataclass
class AgentSayEvent(PluginBaseEvent):
    """Event emitted when the agent wants to say something."""

    type: str = field(default="agent.say", init=False)
    text: str = ""
    user_id: Optional[str] = None  # type: ignore[assignment]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.text:
            raise ValueError("Agent say text cannot be empty")


@dataclass
class AgentSayStartedEvent(PluginBaseEvent):
    """Event emitted when agent speech synthesis starts."""

    type: str = field(default="agent.say_started", init=False)
    text: str = ""
    user_id: Optional[str] = None  # type: ignore[assignment]
    synthesis_id: Optional[str] = None


@dataclass
class AgentSayCompletedEvent(PluginBaseEvent):
    """Event emitted when agent speech synthesis completes."""

    type: str = field(default="agent.say_completed", init=False)
    text: str = ""
    user_id: Optional[str] = None  # type: ignore[assignment]
    synthesis_id: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class AgentSayErrorEvent(PluginBaseEvent):
    """Event emitted when agent speech synthesis encounters an error."""

    type: str = field(default="agent.say_error", init=False)
    text: str = ""
    user_id: Optional[str] = None  # type: ignore[assignment]
    error: Optional[Exception] = None

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"
