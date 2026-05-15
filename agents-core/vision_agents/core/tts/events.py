import uuid
from dataclasses import dataclass, field
from typing import Optional

from getstream.video.rtc import PcmData
from vision_agents.core.events import PluginBaseEvent


@dataclass
class TTSAudioEvent(PluginBaseEvent):
    """Event emitted when TTS audio data is available."""

    type: str = field(default="plugin.tts_audio", init=False)
    data: Optional[PcmData] = None
    chunk_index: int = 0
    is_final_chunk: bool = True
    text_source: Optional[str] = None
    synthesis_id: Optional[str] = None
    epoch: int = 0


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
