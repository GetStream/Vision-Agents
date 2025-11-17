from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class AWSStreamEvent(PluginBaseEvent):
    """Event emitted when AWS provides a stream event."""

    type: str = field(default="plugin.aws.stream", init=False)
    event_data: Optional[Any] = None
