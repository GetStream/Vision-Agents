import uuid
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from types import FunctionType
from dataclasses_json import DataClassJsonMixin

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant


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
class BaseEvent(DataClassJsonMixin):
    """Base class for all events."""

    type: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    participant: Optional[Participant] = None
    # TODO: this is ugly, review why we have this
    user_metadata: Optional[Any] = None

    def user_id(self) -> Optional[str]:
        if self.participant is None:
            return None
        return getattr(self.participant, "user_id")


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

    type: str = field(default="plugin.closed", init=False)
    plugin_type: Optional[str] = None  # "STT", "STS", "VAD"
    provider: Optional[str] = None
    reason: Optional[str] = None
    cleanup_successful: bool = True


@dataclass
class PluginErrorEvent(PluginBaseEvent):
    """Event emitted when a generic plugin error occurs."""

    type: str = field(default="plugin.error", init=False)
    plugin_type: Optional[str] = None  # "STT", "TTS", "STS", "VAD"
    provider: Optional[str] = None
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    is_fatal: bool = False

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclasses.dataclass
class ExceptionEvent:
    exc: Exception
    handler: FunctionType
    type: str = "base.exception"


@dataclasses.dataclass
class HealthCheckEvent(DataClassJsonMixin):
    connection_id: str
    created_at: int
    custom: dict
    type: str = "health.check"


@dataclass
class ConnectionOkEvent(BaseEvent):
    """Event emitted when WebSocket connection is established."""

    type: str = field(default="connection.ok", init=False)
    connection_id: Optional[str] = None
    server_time: Optional[str] = None
    api_key: Optional[str] = None
    user_id: Optional[str] = None  # type: ignore[assignment]


@dataclass
class ConnectionErrorEvent(BaseEvent):
    """Event emitted when WebSocket connection encounters an error."""

    type: str = field(default="connection.error", init=False)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    reconnect_attempt: Optional[int] = None


@dataclass
class ConnectionClosedEvent(BaseEvent):
    """Event emitted when WebSocket connection is closed."""

    type: str = field(default="connection.closed", init=False)
    code: Optional[int] = None
    reason: Optional[str] = None
    was_clean: bool = False
