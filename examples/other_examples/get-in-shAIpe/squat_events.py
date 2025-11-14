"""Events for squat counter processor."""

from dataclasses import dataclass, field
from typing import Optional

from vision_agents.core.events.base import PluginBaseEvent


@dataclass
class SquatCompletedEvent(PluginBaseEvent):
    """
    Event emitted when a squat is completed.
    
    Attributes:
        rep_count: The total number of squats completed
        knee_angle: The knee angle at completion (in degrees)
        timestamp: The timestamp when the squat was completed
    """
    
    type: str = field(default="plugin.squat_completed", init=False)
    rep_count: int = 0
    knee_angle: float = 0.0
    timestamp: float = 0.0

