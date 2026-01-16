import abc
import dataclasses
from dataclasses import dataclass, field
from typing import Iterable


class _Metric(abc.ABC):
    def __init__(self, description: str = "") -> None:
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    @abc.abstractmethod
    def value(self) -> int | float | None: ...

    def __repr__(self):
        return f"<{self.__class__.__name__} value={self.value()}>"


class Counter(_Metric):
    def __init__(self, description: str = "") -> None:
        super(Counter, self).__init__(description)
        self._total = 0

    def inc(self, value: int) -> None:
        self._total += value

    def value(self) -> int:
        return self._total


class Average(_Metric):
    def __init__(self, description: str = "") -> None:
        super(Average, self).__init__(description)
        self._total: int = 0
        self._sum: int | float = 0

    def update(self, value: float | int) -> None:
        self._total += 1
        self._sum += value

    def value(self) -> float | None:
        if not self._total:
            return None

        return self._sum / self._total


@dataclass(frozen=True)
class AgentMetrics:
    """
    Metrics aggregate over a single Agent call.
    """

    # STT Metrics
    stt_latency_ms__avg: Average = field(
        default_factory=lambda: Average("Average STT processing latency")
    )
    stt_audio_duration_ms__total: Counter = field(
        default_factory=lambda: Counter("Duration of audio processed by STT")
    )
    # TTS Metrics
    tts_latency_ms__avg: Average = field(
        default_factory=lambda: Average("TTS synthesis latency")
    )
    tts_audio_duration_ms__total: Counter = field(
        default_factory=lambda: Counter("Duration of synthesized audio")
    )
    tts_characters__total: Counter = field(
        default_factory=lambda: Counter("Characters synthesized by TTS")
    )

    # LLM Metrics
    llm_latency_ms__avg: Average = field(
        default_factory=lambda: Average(
            "LLM response latency (request to complete response)"
        )
    )
    llm_time_to_first_token_ms__avg: Average = field(
        default_factory=lambda: Average("Average LLM time to first token (streaming)")
    )
    llm_input_tokens__total: Counter = field(
        default_factory=lambda: Counter("LLM input/prompt tokens consumed")
    )
    llm_output_tokens__total: Counter = field(
        default_factory=lambda: Counter("LLM output/completion tokens generated")
    )
    llm_tool_calls__total: Counter = field(
        default_factory=lambda: Counter("LLM tool/function calls executed")
    )
    llm_tool_latency_ms__avg: Average = field(
        default_factory=lambda: Average("Average LLM tool execution latency")
    )

    # Turn Detection Metrics
    turn_duration_ms__avg: Average = field(
        default_factory=lambda: Average("Average duration of detected turns")
    )
    turn_trailing_silence_ms__avg: Average = field(
        default_factory=lambda: Average(
            "Average trailing silence duration before turn end"
        )
    )

    # Realtime LLM Metrics
    realtime_audio_input_bytes__total: Counter = field(
        default_factory=lambda: Counter("Audio bytes sent to realtime LLM")
    )
    realtime_audio_output_bytes__total: Counter = field(
        default_factory=lambda: Counter("Audio bytes received from realtime LLM")
    )
    realtime_audio_input_duration_ms__total: Counter = field(
        default_factory=lambda: Counter("Audio duration sent to realtime LLM")
    )
    realtime_audio_output_duration_ms__total: Counter = field(
        default_factory=lambda: Counter("Audio duration received from realtime LLM")
    )
    realtime_user_transcriptions__total: Counter = field(
        default_factory=lambda: Counter("User speech transcriptions from realtime LLM")
    )
    realtime_agent_transcriptions__total: Counter = field(
        default_factory=lambda: Counter("Agent speech transcriptions from realtime LLM")
    )

    # VLM / Vision Metrics
    vlm_inference_latency_ms__avg: Average = field(
        default_factory=lambda: Average("Average VLM inference latency")
    )
    vlm_inferences__total: Counter = field(
        default_factory=lambda: Counter("VLM inference requests")
    )
    vlm_input_tokens__total: Counter = field(
        default_factory=lambda: Counter("VLM input tokens (text + image)")
    )
    vlm_output_tokens__total: Counter = field(
        default_factory=lambda: Counter("VLM output tokens")
    )

    # Video Processor Metrics
    video_frames_processed__total: Counter = field(
        default_factory=lambda: Counter("Video frames processed")
    )
    video_processing_latency_ms__avg: Average = field(
        default_factory=lambda: Average("Average video frame processing latency")
    )

    def to_dict(self, fields: Iterable[str] = ()) -> dict[str, int | float | None]:
        """
        Convert metrics into a dictionary {<metric>: <value>}.

        Args:
            fields: optional list of fields to extract. If empty, extract all fields.

        Returns:
            a dictionary {<metric>: <value>}

        """
        all_fields = dataclasses.asdict(self)
        result = {}
        fields = fields or list(all_fields.keys())

        for field_name in fields:
            field_ = all_fields.get(field_name)
            if field_ is None:
                raise ValueError(f"Unknown field: {field_name}")
            result[field_name] = field_.value()
        return result
