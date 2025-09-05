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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from getstream.video.rtc.track_util import PcmData


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

    # Realtime Events (formerly STS)
    REALTIME_CONNECTED = "realtime_connected"
    REALTIME_DISCONNECTED = "realtime_disconnected"
    REALTIME_AUDIO_INPUT = "realtime_audio_input"
    REALTIME_AUDIO_OUTPUT = "realtime_audio_output"
    REALTIME_TRANSCRIPT = "realtime_transcript"
    REALTIME_RESPONSE = "realtime_response"
    REALTIME_ERROR = "realtime_error"
    REALTIME_CONVERSATION_ITEM = "realtime_conversation_item"

    # VAD Events
    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    VAD_AUDIO = "vad_audio"
    VAD_PARTIAL = "vad_partial"
    VAD_INFERENCE = "vad_inference"
    VAD_ERROR = "vad_error"

    # Generic Plugin Events
    PLUGIN_INITIALIZED = "plugin_initialized"
    PLUGIN_CLOSED = "plugin_closed"
    PLUGIN_ERROR = "plugin_error"

    # call ws events (NOTE: we could process with call_ as similar but then we need to remamp it)
    CALL_MEMBER_ADDED = 'call_member_added'
    CALL_MEMBER_REMOVED = 'call_member_removed'

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
class BaseEvent:
    """Base class for all plugin events."""
    # NOTE: potentially we could use event class name for event name
    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {}


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


@dataclass
class PluginBaseEvent(BaseEvent):
    plugin_name: str | None = None
    plugin_version: str | None = None


@dataclass
class ConnectionBaseEvent(BaseEvent):
    pass


@dataclass
class CallBaseEvent(BaseEvent):
    pass


# ============================================================================
# STT (Speech-to-Text) Events
# ============================================================================


@dataclass
class STTTranscriptEvent(PluginBaseEvent):
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
class STTPartialTranscriptEvent(PluginBaseEvent):
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
class STTErrorEvent(PluginBaseEvent):
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
class STTConnectionEvent(PluginBaseEvent):
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
class TTSAudioEvent(PluginBaseEvent):
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
class TTSSynthesisStartEvent(PluginBaseEvent):
    """Event emitted when TTS synthesis begins."""

    event_type: EventType = field(default=EventType.TTS_SYNTHESIS_START, init=False)
    text: Optional[str] = None
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: Optional[str] = None
    voice_id: Optional[str] = None
    estimated_duration_ms: Optional[float] = None


@dataclass
class TTSSynthesisCompleteEvent(PluginBaseEvent):
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
class TTSErrorEvent(PluginBaseEvent):
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
class TTSConnectionEvent(PluginBaseEvent):
    """Event emitted for TTS connection state changes."""

    event_type: EventType = field(default=EventType.TTS_CONNECTION, init=False)
    connection_state: Optional[ConnectionState] = None
    provider: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


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
    event_type: EventType = field(default=EventType.REALTIME_TRANSCRIPT, init=False)
    text: Optional[str] = None
    is_user: bool = True
    confidence: Optional[float] = None
    conversation_item_id: Optional[str] = None


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
# VAD (Voice Activity Detection) Events
# ============================================================================


@dataclass
class VADSpeechStartEvent(PluginBaseEvent):
    """Event emitted when speech begins."""

    event_type: EventType = field(default=EventType.VAD_SPEECH_START, init=False)
    speech_probability: float = 0.0
    activation_threshold: float = 0.0
    frame_count: int = 1
    audio_data: PcmData = None


@dataclass
class VADSpeechEndEvent(PluginBaseEvent):
    """Event emitted when speech ends."""

    event_type: EventType = field(default=EventType.VAD_SPEECH_END, init=False)
    speech_probability: float = 0.0
    deactivation_threshold: float = 0.0
    total_speech_duration_ms: float = 0.0
    total_frames: int = 0


@dataclass
class VADAudioEvent(PluginBaseEvent):
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
class VADPartialEvent(PluginBaseEvent):
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
class VADInferenceEvent(PluginBaseEvent):
    """Event emitted after each VAD inference window."""

    event_type: EventType = field(default=EventType.VAD_INFERENCE, init=False)
    speech_probability: float = 0.0
    inference_time_ms: float = 0.0
    window_samples: int = 0
    model_rate: int = 16000
    real_time_factor: float = 0.0
    is_speech_active: bool = False
    accumulated_speech_duration_ms: float = 0.0
    accumulated_silence_duration_ms: float = 0.0


@dataclass
class VADErrorEvent(PluginBaseEvent):
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
class PluginInitializedEvent(PluginBaseEvent):
    """Event emitted when a plugin is successfully initialized."""

    event_type: EventType = field(default=EventType.PLUGIN_INITIALIZED, init=False)
    plugin_type: Optional[str] = None  # "STT", "TTS", "STS", "VAD"
    provider: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None


@dataclass
class PluginClosedEvent(PluginBaseEvent):
    """Event emitted when a plugin is closed."""

    event_type: EventType = field(default=EventType.PLUGIN_CLOSED, init=False)
    plugin_type: Optional[str] = None  # "STT", "STS", "VAD"
    provider: Optional[str] = None
    reason: Optional[str] = None
    cleanup_successful: bool = True


@dataclass
class PluginErrorEvent(PluginBaseEvent):
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

# ==
# Call events
# ==

@dataclass
class CallMemberAddedEvent(CallBaseEvent):
    event_type: EventType = field(default=EventType.CALL_MEMBER_ADDED, init=False)


@dataclass
class CallMemberRemovedEvent(CallBaseEvent):
    event_type: EventType = field(default=EventType.CALL_MEMBER_ADDED, init=False)


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
    EventType.REALTIME_CONNECTED: RealtimeConnectedEvent,
    EventType.REALTIME_DISCONNECTED: RealtimeDisconnectedEvent,
    EventType.REALTIME_AUDIO_INPUT: RealtimeAudioInputEvent,
    EventType.REALTIME_AUDIO_OUTPUT: RealtimeAudioOutputEvent,
    EventType.REALTIME_TRANSCRIPT: RealtimeTranscriptEvent,
    EventType.REALTIME_RESPONSE: RealtimeResponseEvent,
    EventType.REALTIME_CONVERSATION_ITEM: RealtimeConversationItemEvent,
    EventType.REALTIME_ERROR: RealtimeErrorEvent,
    EventType.VAD_SPEECH_START: VADSpeechStartEvent,
    EventType.VAD_SPEECH_END: VADSpeechEndEvent,
    EventType.VAD_AUDIO: VADAudioEvent,
    EventType.VAD_PARTIAL: VADPartialEvent,
    EventType.VAD_INFERENCE: VADInferenceEvent,
    EventType.VAD_ERROR: VADErrorEvent,
    EventType.PLUGIN_INITIALIZED: PluginInitializedEvent,
    EventType.PLUGIN_CLOSED: PluginClosedEvent,
    EventType.PLUGIN_ERROR: PluginErrorEvent,
    EventType.CALL_MEMBER_ADDED: CallMemberAddedEvent,
    EventType.CALL_MEMBER_REMOVED: CallMemberRemovedEvent
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
    "PluginBaseEvent",
    "CallBaseEvent",
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
    # Realtime Events
    "RealtimeConnectedEvent",
    "RealtimeDisconnectedEvent",
    "RealtimeAudioInputEvent",
    "RealtimeAudioOutputEvent",
    "RealtimeTranscriptEvent",
    "RealtimeResponseEvent",
    "RealtimeConversationItemEvent",
    "RealtimeErrorEvent",
    # VAD Events
    "VADSpeechStartEvent",
    "VADSpeechEndEvent",
    "VADAudioEvent",
    "VADPartialEvent",
    "VADInferenceEvent",
    "VADErrorEvent",
    # Generic Events
    "PluginInitializedEvent",
    "PluginClosedEvent",
    "PluginErrorEvent",
    # Call Events
    "CallMemberAddedEvent",
    "CallMemberRemovedEvent",
    # Utilities
    "EVENT_CLASS_MAP",
    "create_event",
]
