from vision_agents.core.events import PluginBaseEvent
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class STTErrorEvent(PluginBaseEvent):
    """Event emitted when an STT error occurs."""

    type: str = field(default="plugin.stt_error", init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class STTConnectedEvent(PluginBaseEvent):
    """Event emitted when an STT connection is established."""

    type: str = field(default="plugin.stt_connected", init=False)


@dataclass
class STTDisconnectedEvent(PluginBaseEvent):
    """Event emitted when an STT connection is closed."""

    type: str = field(default="plugin.stt_disconnected", init=False)
    reason: Optional[str] = None
    clean: bool = True
