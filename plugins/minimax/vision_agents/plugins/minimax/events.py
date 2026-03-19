from dataclasses import dataclass, field
from typing import Optional

from vision_agents.core.events import PluginBaseEvent


@dataclass
class MiniMaxStreamEvent(PluginBaseEvent):
    """Event emitted when MiniMax provides a stream chunk."""

    type: str = field(default="plugin.minimax.stream", init=False)
    event_data: Optional[object] = None


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when an error occurs during LLM interaction."""

    type: str = field(default="plugin.minimax.error", init=False)
    error_message: str = ""
    event_data: Optional[object] = None
