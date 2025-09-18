"""
Structured event classes for all GetStream AI plugins.

This module provides type-safe, structured event classes for STT, TTS, STS, VAD,
and other AI plugin types. These events ensure consistency across implementations
and provide better debugging, logging, and integration capabilities.

Key Features:
- Type safety with dataclasses
- Automatic timestamps and unique IDs
- Consistent metadata structure
- Extensible design for future plugin types
- Rich debugging information
"""

import uuid
import dataclasses
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from getstream.video.rtc.track_util import PcmData


class EventType(Enum):
    """Enumeration of all event types across plugin systems."""

    # Realtime Events (formerly STS)
    REALTIME_CONNECTED = "realtime_connected"
    REALTIME_DISCONNECTED = "realtime_disconnected"
    REALTIME_AUDIO_INPUT = "realtime_audio_input"
    REALTIME_AUDIO_OUTPUT = "realtime_audio_output"
    REALTIME_TRANSCRIPT = "realtime_transcript"
    REALTIME_PARTIAL_TRANSCRIPT = "realtime_partial_transcript"
    REALTIME_RESPONSE = "realtime_response"
    REALTIME_ERROR = "realtime_error"
    REALTIME_CONVERSATION_ITEM = "realtime_conversation_item"

    # ... could be same call events
    # connection SFU events (should we have big event type class?)
    # TODO: should we have connection namespace (?) like CONNNECTION_PARTICIPANT_JOINED (?) too long
    PARTICIPANT_JOINED = 'connection_participant_joined'
    PARTICIPANT_LEFT = 'connection_participant_left'


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
    user_metadata: Optional[Dict[str, Any]] = None


@dataclass
class PluginBaseEvent(BaseEvent):
    plugin_name: str | None = None
    plugin_version: str | None = None


@dataclass
class ConnectionBaseEvent(BaseEvent):
    pass


# ============================================================================
# Realtime (Speech-to-Speech) Events
# ============================================================================


@dataclass
class RealtimeConnectedEvent(PluginBaseEvent):
    """Event emitted when realtime connection is established."""
    event_type: EventType = field(default=EventType.REALTIME_CONNECTED, init=False)
    provider: Optional[str] = None
    session_config: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None


@dataclass
class RealtimeDisconnectedEvent(PluginBaseEvent):
    event_type: EventType = field(default=EventType.REALTIME_DISCONNECTED, init=False)
    provider: Optional[str] = None
    reason: Optional[str] = None
    was_clean: bool = True


@dataclass
class RealtimeAudioInputEvent(PluginBaseEvent):
    """Event emitted when audio input is sent to realtime session."""
    event_type: EventType = field(default=EventType.REALTIME_AUDIO_INPUT, init=False)
    audio_data: Optional[bytes] = None
    audio_format: AudioFormat = AudioFormat.PCM_S16
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class RealtimeAudioOutputEvent(PluginBaseEvent):
    """Event emitted when audio output is received from realtime session."""
    event_type: EventType = field(default=EventType.REALTIME_AUDIO_OUTPUT, init=False)
    audio_data: Optional[bytes] = None
    audio_format: AudioFormat = AudioFormat.PCM_S16
    sample_rate: int = 16000
    channels: int = 1
    response_id: Optional[str] = None


@dataclass
class RealtimeTranscriptEvent(PluginBaseEvent):
    """Event emitted when realtime session provides a transcript."""
    original: Optional[Any] = None
    event_type: EventType = field(default=EventType.REALTIME_TRANSCRIPT, init=False)
    text: Optional[str] = None
    user_metadata: Optional[Any] = None


@dataclass
class RealtimePartialTranscriptEvent(BaseEvent):
    original: Optional[any] = None
    event_type: EventType = field(default=EventType.REALTIME_PARTIAL_TRANSCRIPT, init=False)
    text: Optional[str] = None
    user_metadata: Optional[Any] = None


@dataclass
class RealtimeResponseEvent(PluginBaseEvent):
    """Event emitted when realtime session provides a response."""
    event_type: EventType = field(default=EventType.REALTIME_RESPONSE, init=False)
    text: Optional[str] = None
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_complete: bool = True
    conversation_item_id: Optional[str] = None


@dataclass
class RealtimeConversationItemEvent(PluginBaseEvent):
    """Event emitted for conversation item updates in realtime session."""
    event_type: EventType = field(
        default=EventType.REALTIME_CONVERSATION_ITEM, init=False
    )
    item_id: Optional[str] = None
    item_type: Optional[str] = (
        None  # "message", "function_call", "function_call_output"
    )
    status: Optional[str] = None  # "completed", "in_progress", "incomplete"
    role: Optional[str] = None  # "user", "assistant", "system"
    content: Optional[List[Dict[str, Any]]] = None


@dataclass
class RealtimeErrorEvent(PluginBaseEvent):
    """Event emitted when a realtime error occurs."""
    event_type: EventType = field(default=EventType.REALTIME_ERROR, init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


# ============================================================================
# Generic Plugin Events
# ============================================================================


@dataclass
class PluginInitializedEvent(PluginBaseEvent):
    """Event emitted when a plugin is successfully initialized."""

    type: str = field(default="plugin.initialized", init=False)
    plugin_type: Optional[str] = None  # "STT", "TTS", "STS", "VAD"
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

