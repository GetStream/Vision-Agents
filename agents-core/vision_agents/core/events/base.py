import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import FunctionType
from typing import Any, Optional

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


@dataclasses.dataclass
class ExceptionEvent:
    exc: Exception
    handler: FunctionType
    type: str = "base.exception"


@dataclass
class VideoProcessorDetectionEvent(PluginBaseEvent):
    """Base event for video processor detection results.

    Video processor plugins (roboflow, ultralytics, etc.) should inherit from
    this to enable metrics collection.
    """

    type: str = field(default="plugin.video_processor.detection", init=False)
    model_id: Optional[str] = None
    """Identifier of the model used for detection."""
    inference_time_ms: Optional[float] = None
    """Time taken for inference in milliseconds."""
    detection_count: int = 0
    """Number of objects/items detected."""
