import abc
import dataclasses
from dataclasses import dataclass, field
from typing import Iterable, Literal


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

    def load(self, average: float, count: int) -> None:
        """Replace the running average with a known average and sample count."""
        if count <= 0:
            return
        self._total = count
        self._sum = average * count

    @property
    def count(self) -> int:
        """Number of samples contributing to the average."""
        return self._total

    def value(self) -> float | None:
        if not self._total:
            return None

        return self._sum / self._total


MetricsMode = Literal["pipeline", "realtime", "hybrid"]


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
    stt_errors__total: Counter = field(default_factory=lambda: Counter("STT errors"))

    # TTS Metrics
    tts_latency_ms__avg: Average = field(
        default_factory=lambda: Average(
            "TTS total synthesis latency (request to complete)"
        )
    )
    tts_time_to_first_audio_ms__avg: Average = field(
        default_factory=lambda: Average("TTS time to first audio chunk")
    )
    tts_audio_duration_ms__total: Counter = field(
        default_factory=lambda: Counter("Duration of synthesized audio")
    )
    tts_characters__total: Counter = field(
        default_factory=lambda: Counter("Characters synthesized by TTS")
    )
    tts_errors__total: Counter = field(default_factory=lambda: Counter("TTS errors"))

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
    llm_errors__total: Counter = field(default_factory=lambda: Counter("LLM errors"))

    # Turn Detection Metrics
    turns__total: Counter = field(
        default_factory=lambda: Counter("Conversational turns detected")
    )
    turn_duration_ms__avg: Average = field(
        default_factory=lambda: Average("Average duration of detected turns")
    )
    turn_trailing_silence_ms__avg: Average = field(
        default_factory=lambda: Average(
            "Average trailing silence duration before turn end"
        )
    )

    # Realtime LLM Metrics
    realtime_time_to_first_audio_ms__avg: Average = field(
        default_factory=lambda: Average(
            "Realtime time to first audio (speech end to first output)"
        )
    )
    realtime_session_duration_ms__avg: Average = field(
        default_factory=lambda: Average("Average realtime session duration")
    )
    realtime_responses__total: Counter = field(
        default_factory=lambda: Counter("Realtime LLM responses completed")
    )
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
    realtime_errors__total: Counter = field(
        default_factory=lambda: Counter("Realtime LLM errors")
    )

    # VLM / Vision Metrics
    vlm_inference_latency_ms__avg: Average = field(
        default_factory=lambda: Average("Average VLM inference latency")
    )
    vlm_time_to_first_token_ms__avg: Average = field(
        default_factory=lambda: Average("Average VLM time to first token (streaming)")
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
    vlm_errors__total: Counter = field(default_factory=lambda: Counter("VLM errors"))

    # Video Processor Metrics
    video_frames_processed__total: Counter = field(
        default_factory=lambda: Counter("Video frames processed")
    )
    video_processing_latency_ms__avg: Average = field(
        default_factory=lambda: Average("Average video frame processing latency")
    )

    @staticmethod
    def count_field_name(avg_field_name: str) -> str:
        """Map an ``__avg`` field name to its companion ``__count`` key."""
        if not avg_field_name.endswith("__avg"):
            raise ValueError(f"Not an average field: {avg_field_name}")
        return avg_field_name[: -len("__avg")] + "__count"

    @classmethod
    def from_dict(cls, data: dict[str, int | float | None]) -> "AgentMetrics":
        """Reconstruct metrics from a flat dictionary of values.

        Args:
            data: mapping of metric name to its scalar value. Companion
                ``__count`` keys restore average sample counts when present.
        """
        metrics = cls()
        for f in dataclasses.fields(metrics):
            value = data.get(f.name)
            if value is None:
                continue
            metric = getattr(metrics, f.name)
            if isinstance(metric, Counter):
                metric.inc(int(value))
            elif isinstance(metric, Average):
                count = data.get(cls.count_field_name(f.name))
                if isinstance(count, (int, float)) and count > 0:
                    metric.load(float(value), int(count))
                else:
                    metric.update(value)
        return metrics

    def to_dict(self, fields: Iterable[str] = ()) -> dict[str, int | float | None]:
        """Convert metrics into a flat dictionary.

        Every included ``__avg`` field also emits a companion ``__count`` key.

        Args:
            fields: optional list of fields to extract. If empty, extract all
                metric fields plus companion counts for averages.

        Returns:
            a dictionary {<metric>: <value>}
        """
        field_by_name = {f.name: f for f in dataclasses.fields(self)}
        requested = list(fields) if fields else list(field_by_name)

        result: dict[str, int | float | None] = {}
        for field_name in requested:
            if field_name.endswith("__count"):
                avg_name = field_name[: -len("__count")] + "__avg"
                if avg_name not in field_by_name:
                    raise ValueError(f"Unknown field: {field_name}")
                avg_metric = getattr(self, avg_name)
                if not isinstance(avg_metric, Average):
                    raise ValueError(f"Unknown field: {field_name}")
                result[field_name] = avg_metric.count
                continue

            if field_name not in field_by_name:
                raise ValueError(f"Unknown field: {field_name}")
            metric = getattr(self, field_name)
            result[field_name] = metric.value()
            if isinstance(metric, Average):
                result[self.count_field_name(field_name)] = metric.count

        return result

    def infer_mode(self) -> MetricsMode:
        """Infer whether this session used pipeline, realtime, or both."""
        has_pipeline = (
            self.llm_latency_ms__avg.count > 0
            or self.stt_latency_ms__avg.count > 0
            or self.tts_latency_ms__avg.count > 0
            or self.tts_time_to_first_audio_ms__avg.count > 0
        )
        has_realtime = (
            self.realtime_responses__total.value() > 0
            or self.realtime_time_to_first_audio_ms__avg.count > 0
            or self.realtime_audio_output_duration_ms__total.value() > 0
            or self.realtime_user_transcriptions__total.value() > 0
        )
        if has_pipeline and has_realtime:
            return "hybrid"
        if has_realtime:
            return "realtime"
        return "pipeline"
