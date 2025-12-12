from dataclasses import dataclass, field
from typing import Any, Optional

from vision_agents.core.events import PluginBaseEvent


@dataclass
class HuggingFaceStreamEvent(PluginBaseEvent):
    """Event emitted when HuggingFace provides a stream event."""

    type: str = field(default="plugin.huggingface.stream", init=False)
    event_type: Optional[str] = None
    event_data: Optional[Any] = None


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when an LLM encounters an error."""

    type: str = field(default="plugin.llm.error", init=False)
    error_message: Optional[str] = None
    event_data: Optional[Any] = None
