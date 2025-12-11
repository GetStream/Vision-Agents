from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when an LLM encounters an error."""

    type: str = field(default="plugin.llm.error", init=False)
    plugin_name: str = ""
    error_message: Optional[str] = None
    event_data: Optional[Any] = None
