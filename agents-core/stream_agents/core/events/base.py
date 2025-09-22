import uuid
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from getstream.video.rtc.track_util import PcmData


class ConnectionState(Enum):
    """Connection states for streaming plugins."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class AudioFormat(Enum):
    """Supported audio formats."""

    PCM_S16 = "s16"
    PCM_F32 = "f32"
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"


@dataclass
class BaseEvent:
    """Base class for all events."""
    type: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_metadata: Optional[Dict[str, Any]] = None


@dataclass
class PluginBaseEvent(BaseEvent):
    plugin_name: str | None = None
    plugin_version: str | None = None


@dataclass
class PluginInitializedEvent(PluginBaseEvent):
    """Event emitted when a plugin is successfully initialized."""

    type: str = field(default="plugin.initialized", init=False)
    plugin_type: Optional[str] = None
    provider: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None


@dataclass
class PluginClosedEvent(PluginBaseEvent):
    """Event emitted when a plugin is closed."""

    type: EventType = field(default="plugin.closed", init=False)
    plugin_type: Optional[str] = None  # "STT", "STS", "VAD"
    provider: Optional[str] = None
    reason: Optional[str] = None
    cleanup_successful: bool = True


@dataclass
class PluginErrorEvent(PluginBaseEvent):
    """Event emitted when a generic plugin error occurs."""

    type: EventType = field(default="plugin.error", init=False)
    plugin_type: Optional[str] = None  # "STT", "TTS", "STS", "VAD"
    provider: Optional[str] = None
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    is_fatal: bool = False

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"

