from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class OpenAIStreamEvent(PluginBaseEvent):
    """Event emitted when OpenAI provides a stream event."""

    type: str = field(default="plugin.openai.stream", init=False)
    event_type: Optional[str] = None
    event_data: Optional[Any] = None


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when an LLM encounters an error."""

    type: str = field(default="plugin.llm.error", init=False)
    error_message: Optional[str] = None
    event_data: Optional[Any] = None
