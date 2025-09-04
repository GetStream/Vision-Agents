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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Enumeration of all event types across plugin systems."""

    # STT Events
    STT_TRANSCRIPT = "stt_transcript"
    STT_PARTIAL_TRANSCRIPT = "stt_partial_transcript"
    STT_ERROR = "stt_error"
    STT_CONNECTION = "stt_connection"

    # TTS Events
    TTS_AUDIO = "tts_audio"
    TTS_SYNTHESIS_START = "tts_synthesis_start"
    TTS_SYNTHESIS_COMPLETE = "tts_synthesis_complete"
    TTS_ERROR = "tts_error"
    TTS_CONNECTION = "tts_connection"

    # STS Events
    STS_CONNECTED = "sts_connected"
    STS_DISCONNECTED = "sts_disconnected"
    STS_AUDIO_INPUT = "sts_audio_input"
    STS_AUDIO_OUTPUT = "sts_audio_output"
    STS_TRANSCRIPT = "sts_transcript"
    STS_RESPONSE = "sts_response"
    STS_ERROR = "sts_error"
    STS_CONVERSATION_ITEM = "sts_conversation_item"

    # VAD Events
    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    VAD_AUDIO = "vad_audio"
    VAD_PARTIAL = "vad_partial"
    VAD_ERROR = "vad_error"

    # Generic Plugin Events
    PLUGIN_INITIALIZED = "plugin_initialized"
    PLUGIN_CLOSED = "plugin_closed"
    PLUGIN_ERROR = "plugin_error"


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
    """Base class for all plugin events."""

    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_metadata: Optional[Dict[str, Any]] = None
    plugin_name: Optional[str] = None
    plugin_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {}

        import dataclasses

        for field_info in dataclasses.fields(self):
            field_value = getattr(self, field_info.name)
            if isinstance(field_value, (datetime, Enum)):
                result[field_info.name] = (
                    field_value.value
                    if isinstance(field_value, Enum)
                    else str(field_value)
                )
            else:
                result[field_info.name] = field_value
        return result


# ============================================================================
# STT (Speech-to-Text) Events
# ============================================================================


@dataclass
class STTTranscriptEvent(BaseEvent):
    """Event emitted when a complete transcript is available."""

    event_type: EventType = field(default=EventType.STT_TRANSCRIPT, init=False)
    text: str = ""
    confidence: Optional[float] = None
    language: Optional[str] = None
    processing_time_ms: Optional[float] = None
    audio_duration_ms: Optional[float] = None
    model_name: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None
    is_final: bool = True

    def __post_init__(self):
        if not self.text:
            raise ValueError("Transcript text cannot be empty")


@dataclass
class STTPartialTranscriptEvent(BaseEvent):
    """Event emitted when a partial transcript is available."""

    event_type: EventType = field(default=EventType.STT_PARTIAL_TRANSCRIPT, init=False)
    text: str = ""
    confidence: Optional[float] = None
    language: Optional[str] = None
    processing_time_ms: Optional[float] = None
    audio_duration_ms: Optional[float] = None
    model_name: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None
    is_final: bool = False


@dataclass
class STTErrorEvent(BaseEvent):
    """Event emitted when an STT error occurs."""

    event_type: EventType = field(default=EventType.STT_ERROR, init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    retry_count: int = 0
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class STTConnectionEvent(BaseEvent):
    """Event emitted for STT connection state changes."""

    event_type: EventType = field(default=EventType.STT_CONNECTION, init=False)
    connection_state: Optional[ConnectionState] = None
    provider: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    reconnect_attempts: int = 0


# ============================================================================
# TTS (Text-to-Speech) Events
# ============================================================================


@dataclass
class TTSAudioEvent(BaseEvent):
    """Event emitted when TTS audio data is available."""

    event_type: EventType = field(default=EventType.TTS_AUDIO, init=False)
    audio_data: Optional[bytes] = None
    audio_format: AudioFormat = AudioFormat.PCM_S16
    sample_rate: int = 16000
    channels: int = 1
    chunk_index: int = 0
    is_final_chunk: bool = True
    text_source: Optional[str] = None
    synthesis_id: Optional[str] = None


@dataclass
class TTSSynthesisStartEvent(BaseEvent):
    """Event emitted when TTS synthesis begins."""

    event_type: EventType = field(default=EventType.TTS_SYNTHESIS_START, init=False)
    text: Optional[str] = None
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: Optional[str] = None
    voice_id: Optional[str] = None
    estimated_duration_ms: Optional[float] = None


@dataclass
class TTSSynthesisCompleteEvent(BaseEvent):
    """Event emitted when TTS synthesis completes."""

    event_type: EventType = field(default=EventType.TTS_SYNTHESIS_COMPLETE, init=False)
    synthesis_id: Optional[str] = None
    text: Optional[str] = None
    total_audio_bytes: int = 0
    synthesis_time_ms: float = 0.0
    audio_duration_ms: Optional[float] = None
    chunk_count: int = 1
    real_time_factor: Optional[float] = None


@dataclass
class TTSErrorEvent(BaseEvent):
    """Event emitted when a TTS error occurs."""

    event_type: EventType = field(default=EventType.TTS_ERROR, init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    text_source: Optional[str] = None
    synthesis_id: Optional[str] = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class TTSConnectionEvent(BaseEvent):
    """Event emitted for TTS connection state changes."""

    event_type: EventType = field(default=EventType.TTS_CONNECTION, init=False)
    connection_state: Optional[ConnectionState] = None
    provider: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# STS (Speech-to-Speech) Events
# ============================================================================


@dataclass
class STSConnectedEvent(BaseEvent):
    """Event emitted when STS connection is established."""

    event_type: EventType = field(default=EventType.STS_CONNECTED, init=False)
    provider: Optional[str] = None
    session_config: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None


@dataclass
class STSDisconnectedEvent(BaseEvent):
    """Event emitted when STS connection is closed."""

    event_type: EventType = field(default=EventType.STS_DISCONNECTED, init=False)
    provider: Optional[str] = None
    reason: Optional[str] = None
    was_clean: bool = True


@dataclass
class STSAudioInputEvent(BaseEvent):
    """Event emitted when audio input is sent to STS."""

    event_type: EventType = field(default=EventType.STS_AUDIO_INPUT, init=False)
    audio_data: Optional[bytes] = None
    audio_format: AudioFormat = AudioFormat.PCM_S16
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class STSAudioOutputEvent(BaseEvent):
    """Event emitted when audio output is received from STS."""

    event_type: EventType = field(default=EventType.STS_AUDIO_OUTPUT, init=False)
    audio_data: Optional[bytes] = None
    audio_format: AudioFormat = AudioFormat.PCM_S16
    sample_rate: int = 16000
    channels: int = 1
    response_id: Optional[str] = None


@dataclass
class STSTranscriptEvent(BaseEvent):
    """Event emitted when STS provides a transcript."""

    event_type: EventType = field(default=EventType.STS_TRANSCRIPT, init=False)
    text: Optional[str] = None
    is_user: bool = True
    confidence: Optional[float] = None
    conversation_item_id: Optional[str] = None


@dataclass
class STSResponseEvent(BaseEvent):
    """Event emitted when STS provides a response."""

    event_type: EventType = field(default=EventType.STS_RESPONSE, init=False)
    text: Optional[str] = None
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_complete: bool = True
    conversation_item_id: Optional[str] = None


@dataclass
class STSConversationItemEvent(BaseEvent):
    """Event emitted for conversation item updates in STS."""

    event_type: EventType = field(default=EventType.STS_CONVERSATION_ITEM, init=False)
    item_id: Optional[str] = None
    item_type: Optional[str] = (
        None  # "message", "function_call", "function_call_output"
    )
    status: Optional[str] = None  # "completed", "in_progress", "incomplete"
    role: Optional[str] = None  # "user", "assistant", "system"
    content: Optional[List[Dict[str, Any]]] = None


@dataclass
class STSErrorEvent(BaseEvent):
    """Event emitted when an STS error occurs."""

    event_type: EventType = field(default=EventType.STS_ERROR, init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


# ============================================================================
# VAD (Voice Activity Detection) Events
# ============================================================================


@dataclass
class VADSpeechStartEvent(BaseEvent):
    """Event emitted when speech begins."""

    event_type: EventType = field(default=EventType.VAD_SPEECH_START, init=False)
    speech_probability: float = 0.0
    activation_threshold: float = 0.0
    frame_count: int = 1


@dataclass
class VADSpeechEndEvent(BaseEvent):
    """Event emitted when speech ends."""

    event_type: EventType = field(default=EventType.VAD_SPEECH_END, init=False)
    speech_probability: float = 0.0
    deactivation_threshold: float = 0.0
    total_speech_duration_ms: float = 0.0
    total_frames: int = 0


@dataclass
class VADAudioEvent(BaseEvent):
    """Event emitted when VAD detects complete speech segment."""

    event_type: EventType = field(default=EventType.VAD_AUDIO, init=False)
    audio_data: Optional[bytes] = None  # PCM audio data
    sample_rate: int = 16000
    audio_format: AudioFormat = AudioFormat.PCM_S16
    channels: int = 1
    duration_ms: Optional[float] = None
    speech_probability: Optional[float] = None
    frame_count: int = 0


@dataclass
class VADPartialEvent(BaseEvent):
    """Event emitted during ongoing speech detection."""

    event_type: EventType = field(default=EventType.VAD_PARTIAL, init=False)
    audio_data: Optional[bytes] = None  # PCM audio data
    sample_rate: int = 16000
    audio_format: AudioFormat = AudioFormat.PCM_S16
    channels: int = 1
    duration_ms: Optional[float] = None
    speech_probability: Optional[float] = None
    frame_count: int = 0
    is_speech_active: bool = True


@dataclass
class VADErrorEvent(BaseEvent):
    """Event emitted when a VAD error occurs."""

    event_type: EventType = field(default=EventType.VAD_ERROR, init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    frame_data_available: bool = False

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


# ============================================================================
# Generic Plugin Events
# ============================================================================


@dataclass
class PluginInitializedEvent(BaseEvent):
    """Event emitted when a plugin is successfully initialized."""

    event_type: EventType = field(default=EventType.PLUGIN_INITIALIZED, init=False)
    plugin_type: Optional[str] = None  # "STT", "TTS", "STS", "VAD"
    provider: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None


@dataclass
class PluginClosedEvent(BaseEvent):
    """Event emitted when a plugin is closed."""

    event_type: EventType = field(default=EventType.PLUGIN_CLOSED, init=False)
    plugin_type: Optional[str] = None  # "STT", "STS", "VAD"
    provider: Optional[str] = None
    reason: Optional[str] = None
    cleanup_successful: bool = True


@dataclass
class PluginErrorEvent(BaseEvent):
    """Event emitted when a generic plugin error occurs."""

    event_type: EventType = field(default=EventType.PLUGIN_ERROR, init=False)
    plugin_type: Optional[str] = None  # "STT", "TTS", "STS", "VAD"
    provider: Optional[str] = None
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    is_fatal: bool = False

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


# ============================================================================
# Event Type Mappings for Easy Access
# ============================================================================

# Map event types to their corresponding classes
EVENT_CLASS_MAP = {
    EventType.STT_TRANSCRIPT: STTTranscriptEvent,
    EventType.STT_PARTIAL_TRANSCRIPT: STTPartialTranscriptEvent,
    EventType.STT_ERROR: STTErrorEvent,
    EventType.STT_CONNECTION: STTConnectionEvent,
    EventType.TTS_AUDIO: TTSAudioEvent,
    EventType.TTS_SYNTHESIS_START: TTSSynthesisStartEvent,
    EventType.TTS_SYNTHESIS_COMPLETE: TTSSynthesisCompleteEvent,
    EventType.TTS_ERROR: TTSErrorEvent,
    EventType.TTS_CONNECTION: TTSConnectionEvent,
    EventType.STS_CONNECTED: STSConnectedEvent,
    EventType.STS_DISCONNECTED: STSDisconnectedEvent,
    EventType.STS_AUDIO_INPUT: STSAudioInputEvent,
    EventType.STS_AUDIO_OUTPUT: STSAudioOutputEvent,
    EventType.STS_TRANSCRIPT: STSTranscriptEvent,
    EventType.STS_RESPONSE: STSResponseEvent,
    EventType.STS_CONVERSATION_ITEM: STSConversationItemEvent,
    EventType.STS_ERROR: STSErrorEvent,
    EventType.VAD_SPEECH_START: VADSpeechStartEvent,
    EventType.VAD_SPEECH_END: VADSpeechEndEvent,
    EventType.VAD_AUDIO: VADAudioEvent,
    EventType.VAD_PARTIAL: VADPartialEvent,
    EventType.VAD_ERROR: VADErrorEvent,
    EventType.PLUGIN_INITIALIZED: PluginInitializedEvent,
    EventType.PLUGIN_CLOSED: PluginClosedEvent,
    EventType.PLUGIN_ERROR: PluginErrorEvent,
}


def create_event(event_type: EventType, **kwargs) -> BaseEvent:
    """
    Create an event instance of the appropriate type.

    Args:
        event_type: The type of event to create
        **kwargs: Event-specific parameters

    Returns:
        An instance of the appropriate event class

    Raises:
        ValueError: If the event type is not recognized
    """
    if event_type not in EVENT_CLASS_MAP:
        raise ValueError(f"No event class defined for type: {event_type}")

    event_class = EVENT_CLASS_MAP[event_type]
    return event_class(**kwargs)


__all__ = [
    # Enums
    "EventType",
    "ConnectionState",
    "AudioFormat",
    # Base classes
    "BaseEvent",
    # STT Events
    "STTTranscriptEvent",
    "STTPartialTranscriptEvent",
    "STTErrorEvent",
    "STTConnectionEvent",
    # TTS Events
    "TTSAudioEvent",
    "TTSSynthesisStartEvent",
    "TTSSynthesisCompleteEvent",
    "TTSErrorEvent",
    "TTSConnectionEvent",
    # STS Events
    "STSConnectedEvent",
    "STSDisconnectedEvent",
    "STSAudioInputEvent",
    "STSAudioOutputEvent",
    "STSTranscriptEvent",
    "STSResponseEvent",
    "STSConversationItemEvent",
    "STSErrorEvent",
    # VAD Events
    "VADSpeechStartEvent",
    "VADSpeechEndEvent",
    "VADAudioEvent",
    "VADPartialEvent",
    "VADErrorEvent",
    # Generic Events
    "PluginInitializedEvent",
    "PluginClosedEvent",
    "PluginErrorEvent",
    # Utilities
    "EVENT_CLASS_MAP",
    "create_event",
]
