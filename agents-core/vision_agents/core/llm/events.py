from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vision_agents.core.events import PluginBaseEvent


@dataclass
class RealtimeConnectedEvent(PluginBaseEvent):
    """Event emitted when realtime connection is established."""

    type: str = field(default="plugin.realtime_connected", init=False)
    session_id: Optional[str] = None
    session_config: Optional[dict[str, Any]] = None
    capabilities: Optional[list[str]] = None


@dataclass
class RealtimeDisconnectedEvent(PluginBaseEvent):
    """Event emitted when realtime connection is closed."""

    type: str = field(default="plugin.realtime_disconnected", init=False)
    session_id: Optional[str] = None
    reason: Optional[str] = None
    clean: bool = True


@dataclass
class LLMResponseFinalEvent(PluginBaseEvent):
    """Event emitted when a final LLM response is received."""

    type: str = field(default="plugin.llm_response_final", init=False)

    text: str = ""
    """Full LLM response text."""

    model: Optional[str] = None
    """Model being used for this response."""


@dataclass
class ToolStartEvent(PluginBaseEvent):
    """Event emitted when a tool execution starts."""

    type: str = field(default="plugin.llm.tool.start", init=False)
    tool_name: str = ""
    arguments: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolEndEvent(PluginBaseEvent):
    """Event emitted when a tool execution ends."""

    type: str = field(default="plugin.llm.tool.end", init=False)
    tool_name: str = ""
    success: bool = True
    result: Optional[Any] = None
    error: Optional[str] = None
    tool_call_id: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when a non-realtime LLM error occurs."""

    type: str = field(default="plugin.llm_error", init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    request_id: Optional[str] = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"
