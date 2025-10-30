from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class VertexAIResponseEvent(PluginBaseEvent):
    """Event emitted when Vertex AI provides a response chunk."""
    type: str = field(default='plugin.vertex_ai.response', init=False)
    response_chunk: Optional[Any] = None


@dataclass
class VertexAIErrorEvent(PluginBaseEvent):
    """Event emitted when Vertex AI encounters an error."""
    type: str = field(default='plugin.vertex_ai.error', init=False)
    error: Optional[Any] = None
