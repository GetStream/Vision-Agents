import uuid
from dataclasses import dataclass, field
from typing import Optional

from vision_agents.core.events import PluginBaseEvent


@dataclass
class TTSErrorEvent(PluginBaseEvent):
    """Event emitted when a TTS error occurs."""

    type: str = field(default="plugin.tts_error", init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class TTSSynthesisStartEvent(PluginBaseEvent):
    """Event emitted when TTS synthesis begins."""

    type: str = field(default="plugin.tts_synthesis_start", init=False)
    text: Optional[str] = None
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: Optional[str] = None
    voice_id: Optional[str] = None
    estimated_duration_ms: Optional[float] = None


@dataclass
class TTSSynthesisCompleteEvent(PluginBaseEvent):
    """Event emitted when TTS synthesis completes."""

    type: str = field(default="plugin.tts_synthesis_complete", init=False)
    synthesis_id: Optional[str] = None
    text: Optional[str] = None
    total_audio_bytes: int = 0
    synthesis_time_ms: float = 0.0
    audio_duration_ms: Optional[float] = None
    chunk_count: int = 1
    real_time_factor: Optional[float] = None


@dataclass
class TTSConnectedEvent(PluginBaseEvent):
    """Event emitted when a TTS connection is established."""

    type: str = field(default="plugin.tts_connected", init=False)


@dataclass
class TTSDisconnectedEvent(PluginBaseEvent):
    """Event emitted when a TTS connection is closed."""

    type: str = field(default="plugin.tts_disconnected", init=False)
    reason: Optional[str] = None
    clean: bool = True
