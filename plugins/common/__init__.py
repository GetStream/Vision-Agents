"""Common abstract base classes and utilities shared by all Stream AI plugins.

Moving forward, provider wheels (`stream-plugins-deepgram`, …) should depend on
this package for the canonical definitions of STT, TTS, VAD, …
"""

from .stt import STT
from .tts import TTS
from .vad import VAD
from .sts import STS
from .events import (
    EventType,
    ConnectionState,
    AudioFormat,
    BaseEvent,

    # STT Events
    STTTranscriptEvent,
    STTPartialTranscriptEvent,
    STTErrorEvent,
    STTConnectionEvent,

    # TTS Events
    TTSAudioEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
    TTSErrorEvent,
    TTSConnectionEvent,

    # STS Events
    STSConnectedEvent,
    STSDisconnectedEvent,
    STSAudioInputEvent,
    STSAudioOutputEvent,
    STSTranscriptEvent,
    STSResponseEvent,
    STSConversationItemEvent,
    STSErrorEvent,

    # VAD Events
    VADSpeechStartEvent,
    VADSpeechEndEvent,
    VADAudioEvent,
    VADPartialEvent,
    VADErrorEvent,

    # Generic Events
    PluginInitializedEvent,
    PluginClosedEvent,
    PluginErrorEvent,
)
from .event_utils import (
    EventFilter,
    EventRegistry,
    EventLogger,
    register_global_event,
    get_global_registry,
    get_global_logger,
)
from .event_serialization import serialize_event, serialize_events, deserialize_event
from .event_metrics import calculate_stt_metrics, calculate_tts_metrics, calculate_vad_metrics
from .events import create_event

__all__ = [
    # Base classes
    "STT",
    "TTS",
    "VAD",
    "STS",

    # Event system
    "EventType",
    "ConnectionState",
    "AudioFormat",
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

    # Event utilities
    "EventFilter",
    "EventRegistry",
    "EventLogger",
    "register_global_event",
    "get_global_registry",
    "get_global_logger",
    "create_event",

    # Event serialization
    "serialize_event",
    "serialize_events",
    "deserialize_event",

    # Event metrics
    "calculate_stt_metrics",
    "calculate_tts_metrics",
    "calculate_vad_metrics",
]
