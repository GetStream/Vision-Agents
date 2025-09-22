from dataclasses import dataclass, field
from stream_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class XAIChunkEvent(PluginBaseEvent):
    """Event emitted when xAI provides a chunk."""
    type: str = field(default='plugin.xai.chunk', init=False)
    chunk: Optional[Any] = None
