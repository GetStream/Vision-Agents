from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vision_agents.core.agents.transcript.buffer import TranscriptMode
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
class RealtimeAgentSpeechTranscriptionEvent(PluginBaseEvent):
    """Event emitted when agent speech transcription is available from realtime session.

    Args:
        text: The transcript text.
        mode: How to interpret the text:
            - "delta": incremental chunk, more to come
            - "replacement": full utterance so far, more to come
            - "final": utterance complete (text may be empty to just signal finality)
        original: The raw provider event, if available.
    """

    type: str = field(default="plugin.realtime_agent_speech_transcription", init=False)
    text: str = ""
    mode: TranscriptMode = "delta"
    original: Optional[Any] = None


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
