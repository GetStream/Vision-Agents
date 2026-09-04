from dataclasses import dataclass, field
from typing import Optional

from vision_agents.core.edge.call import Call
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import BaseEvent


@dataclass
class AgentFinishEvent(BaseEvent):
    """Event emitted when agent.finish() call ended."""

    type: str = field(default="agent.finish", init=False)


@dataclass
class AgentJoinedCallEvent(BaseEvent):
    """Event emitted after the agent has joined a call."""

    type: str = field(default="agent.joined_call", init=False)
    call: Call = field(kw_only=True)


@dataclass
class AgentLeftCallEvent(BaseEvent):
    """Event emitted when the agent leaves a call."""

    type: str = field(default="agent.left_call", init=False)
    call: Call = field(kw_only=True)


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
class UserTranscriptEvent(BaseEvent):
    """Emitted with the final user transcript that triggers an LLM turn."""

    type: str = field(default="agent.user_transcript", init=False)
    text: str = ""
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
