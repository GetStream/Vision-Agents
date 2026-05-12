from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vision_agents.core.edge.types import Participant
from vision_agents.core.events import BaseEvent, PluginBaseEvent


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
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.text:
            raise ValueError("Agent say text cannot be empty")


@dataclass
class AgentSayStartedEvent(PluginBaseEvent):
    """Event emitted when agent speech synthesis starts."""

    type: str = field(default="agent.say_started", init=False)
    text: str = ""
    synthesis_id: Optional[str] = None


@dataclass
class AgentSayCompletedEvent(PluginBaseEvent):
    """Event emitted when agent speech synthesis completes."""

    type: str = field(default="agent.say_completed", init=False)
    text: str = ""
    synthesis_id: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class AgentSayErrorEvent(PluginBaseEvent):
    """Event emitted when agent speech synthesis encounters an error."""

    type: str = field(default="agent.say_error", init=False)
    text: str = ""
    error: Optional[Exception] = None

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class UserTurnStartedEvent(BaseEvent):
    """Emitted when the user starts speaking."""

    type: str = field(default="agent.user_turn_started", init=False)
    participant: Optional[Participant] = None


@dataclass
class UserTurnEndedEvent(BaseEvent):
    """Emitted when the user stops speaking."""

    type: str = field(default="agent.user_turn_ended", init=False)
    participant: Optional[Participant] = None


@dataclass
class AgentTurnStartedEvent(BaseEvent):
    """Emitted when the agent starts speaking (first audio chunk leaving the pipeline)."""

    type: str = field(default="agent.agent_turn_started", init=False)


@dataclass
class AgentTurnEndedEvent(BaseEvent):
    """Emitted when the agent stops speaking. ``interrupted`` is True for barge-in."""

    type: str = field(default="agent.agent_turn_ended", init=False)
    interrupted: bool = False
