from dataclasses import dataclass, field

from getstream.video.rtc import PcmData

from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any, Dict
import uuid


@dataclass
class RealtimeConnectedEvent(PluginBaseEvent):
    """Event emitted when realtime connection is established."""

    type: str = field(default="plugin.realtime_connected", init=False)
    provider: Optional[str] = None
    session_id: Optional[str] = None
    session_config: Optional[dict[str, Any]] = None
    capabilities: Optional[list[str]] = None


@dataclass
class RealtimeDisconnectedEvent(PluginBaseEvent):
    """Event emitted when realtime connection is closed."""

    type: str = field(default="plugin.realtime_disconnected", init=False)
    provider: Optional[str] = None
    session_id: Optional[str] = None
    reason: Optional[str] = None
    was_clean: bool = True


@dataclass
class RealtimeAudioInputEvent(PluginBaseEvent):
    """Event emitted when audio input is sent to realtime session."""

    type: str = field(default="plugin.realtime_audio_input", init=False)
    data: Optional[PcmData] = None


@dataclass
class RealtimeAudioOutputEvent(PluginBaseEvent):
    """Event emitted when audio output is received from realtime session."""

    type: str = field(default="plugin.realtime_audio_output", init=False)
    data: Optional[PcmData] = None
    response_id: Optional[str] = None


@dataclass
class RealtimeResponseEvent(PluginBaseEvent):
    """Event emitted when realtime session provides a response."""

    type: str = field(default="plugin.realtime_response", init=False)
    original: Optional[str] = None
    text: Optional[str] = None
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_complete: bool = True
    conversation_item_id: Optional[str] = None


@dataclass
class RealtimeConversationItemEvent(PluginBaseEvent):
    """Event emitted for conversation item updates in realtime session."""

    type: str = field(default="plugin.realtime_conversation_item", init=False)
    item_id: Optional[str] = None
    item_type: Optional[str] = (
        None  # "message", "function_call", "function_call_output"
    )
    status: Optional[str] = None  # "completed", "in_progress", "incomplete"
    role: Optional[str] = None  # "user", "assistant", "system"
    content: Optional[list[dict[str, Any]]] = None


@dataclass
class RealtimeErrorEvent(PluginBaseEvent):
    """Event emitted when a realtime error occurs."""

    type: str = field(default="plugin.realtime_error", init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class LLMRequestStartedEvent(PluginBaseEvent):
    """Event emitted when an LLM request begins."""

    type: str = field(default="plugin.llm_request_started", init=False)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this request."""
    model: Optional[str] = None
    """Model being used for this request."""
    streaming: bool = False
    """Whether this is a streaming request."""


@dataclass
class LLMResponseChunkEvent(PluginBaseEvent):
    type: str = field(default="plugin.llm_response_chunk", init=False)
    content_index: int | None = None
    """The index of the content part that the text delta was added to."""

    delta: str | None = None
    """The text delta that was added."""

    item_id: Optional[str] = None
    """The ID of the output item that the text delta was added to."""

    output_index: Optional[int] = None
    """The index of the output item that the text delta was added to."""

    sequence_number: Optional[int] = None
    """The sequence number for this event."""

    # Timing for first chunk detection
    is_first_chunk: bool = False
    """Whether this is the first chunk in the stream."""
    time_to_first_token_ms: Optional[float] = None
    """Time from request start to this first chunk (only set if is_first_chunk=True)."""


@dataclass
class LLMResponseCompletedEvent(PluginBaseEvent):
    """Event emitted after an LLM response is processed."""

    type: str = field(default="plugin.llm_response_completed", init=False)
    original: Any = None
    text: str = ""
    item_id: Optional[str] = None

    # Timing metrics
    latency_ms: Optional[float] = None
    """Total time from request to complete response."""
    time_to_first_token_ms: Optional[float] = None
    """Time from request to first token received (streaming)."""

    # Token usage
    input_tokens: Optional[int] = None
    """Number of input/prompt tokens consumed."""
    output_tokens: Optional[int] = None
    """Number of output/completion tokens generated."""
    total_tokens: Optional[int] = None
    """Total tokens (input + output). May differ from sum if cached."""

    # Model info
    model: Optional[str] = None
    """Model identifier used for this response."""


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
class RealtimeUserSpeechTranscriptionEvent(PluginBaseEvent):
    """Event emitted when user speech transcription is available from realtime session."""

    type: str = field(default="plugin.realtime_user_speech_transcription", init=False)
    text: str = ""
    original: Optional[Any] = None


@dataclass
class RealtimeAgentSpeechTranscriptionEvent(PluginBaseEvent):
    """Event emitted when agent speech transcription is available from realtime session."""

    type: str = field(default="plugin.realtime_agent_speech_transcription", init=False)
    text: str = ""
    original: Optional[Any] = None


# =============================================================================
# VLM / Vision Events
# =============================================================================


@dataclass
class VLMInferenceStartEvent(PluginBaseEvent):
    """Event emitted when a VLM inference starts."""

    type: str = field(default="plugin.vlm_inference_start", init=False)
    inference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: Optional[str] = None
    frames_count: int = 0


@dataclass
class VLMInferenceCompletedEvent(PluginBaseEvent):
    """Event emitted when a VLM inference completes."""

    type: str = field(default="plugin.vlm_inference_completed", init=False)
    inference_id: Optional[str] = None
    model: Optional[str] = None
    text: str = ""

    # Timing metrics
    latency_ms: Optional[float] = None
    """Total time from request to complete response."""

    # Token usage
    input_tokens: Optional[int] = None
    """Number of input tokens (text + image tokens)."""
    output_tokens: Optional[int] = None
    """Number of output tokens generated."""

    # Vision-specific metrics
    frames_processed: int = 0
    """Number of video frames processed in this inference."""
    detections: int = 0
    """Number of objects/items detected (for detection models)."""


@dataclass
class VLMErrorEvent(PluginBaseEvent):
    """Event emitted when a VLM error occurs."""

    type: str = field(default="plugin.vlm_error", init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    inference_id: Optional[str] = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


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
