from vision_agents.core.events import PluginBaseEvent, AudioFormat
from getstream.video.rtc.track_util import PcmData
from dataclasses import field, dataclass
from typing import Optional


@dataclass
class VADSpeechStartEvent(PluginBaseEvent):
    """Event emitted when speech begins."""

    type: str = field(default='plugin.vad_speech_start', init=False)
    speech_probability: float = 0.0
    activation_threshold: float = 0.0
    frame_count: int = 1
    audio_data: PcmData = None


@dataclass
class VADSpeechEndEvent(PluginBaseEvent):
    """Event emitted when speech ends."""

    type: str = field(default='plugin.vad_speech_end', init=False)
    speech_probability: float = 0.0
    deactivation_threshold: float = 0.0
    total_speech_duration_ms: float = 0.0
    total_frames: int = 0


@dataclass
class VADAudioEvent(PluginBaseEvent):
    """Event emitted when VAD detects complete speech segment."""

    type: str = field(default='plugin.vad_audio', init=False)
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

    type: str = field(default='plugin.vad_partial', init=False)
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

    type: str = field(default='plugin.vad_inference', init=False)
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

    type: str = field(default='plugin.vad_error', init=False)
    error: Optional[Exception] = None
    error_code: Optional[str] = None
    context: Optional[str] = None
    frame_data_available: bool = False

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"
