from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class ClaudeStreamEvent(PluginBaseEvent):
    """Event emitted when Claude provides a stream event."""
    type: str = field(default='plugin.anthropic.claude_stream', init=False)
    event_data: Optional[Any] = None
