"""Turn detection events for the new event system."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vision_agents.core.edge.types import Participant
from vision_agents.core.events.base import PluginBaseEvent


@dataclass
class TurnStartedEvent(PluginBaseEvent):
    """
    Event emitted when a speaker starts their turn.

    Attributes:
        speaker_id: ID of the speaker who started speaking
        confidence: Confidence level of the turn detection (0.0-1.0)
        duration: Duration of audio processed (seconds)
        custom: Additional metadata specific to the turn detection implementation
    """

    type: str = field(default="plugin.turn_started", init=False)
    participant: Optional[Participant] = None
    confidence: Optional[float] = None
    custom: Optional[Dict[str, Any]] = None


@dataclass
class TurnEndedEvent(PluginBaseEvent):
    """
    Event emitted when a speaker completes their turn.

    Attributes:
        speaker_id: ID of the speaker who finished speaking
        confidence: Confidence level of the turn completion detection (0.0-1.0)
        duration: Duration of the turn (seconds)
        custom: Additional metadata specific to the turn detection implementation
    """

    type: str = field(default="plugin.turn_ended", init=False)
    participant: Optional[Participant] = None
    confidence: Optional[float] = None
    trailing_silence_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    custom: Optional[Dict[str, Any]] = None
    eager_end_of_turn: bool = False


__all__ = ["TurnStartedEvent", "TurnEndedEvent"]
