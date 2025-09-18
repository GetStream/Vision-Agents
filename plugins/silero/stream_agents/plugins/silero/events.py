from dataclasses import dataclass, field
from stream_agents.core.vad.events import VADSpeechEndEvent, VADAudioEvent, VADPartialEvent


@dataclass
class SileroVADEndEvent(VADSpeechEndEvent):
    """Event emitted when speech ends."""

    type: str = field(default='plugin.silero.vad_speech_end', init=False)
    avg_speech_probability: float = 0.0
    inference_performance_ms: float = 0.0
    model_confidence: float = 0.0


@dataclass
class SileroVADAudioEvent(VADAudioEvent):
    """Event emitted when VAD detects complete speech segment."""

    type: str = field(default='plugin.silero.vad_audio', init=False)
    start_speech_probability: float = 0.0
    end_speech_probability: float = 0.0
    avg_inference_time_ms: float = 0.0
    total_inferences: int = 0
    model_confidence: float = 0.0


@dataclass
class SileroVADPartialEvent(VADPartialEvent):
    """Event emitted during ongoing speech detection."""

    type: str = field(default='plugin.silero.vad_partial', init=False)
    inference_time_ms: float = 0.0
    model_confidence: float = 0.0

