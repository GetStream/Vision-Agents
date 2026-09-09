"""OpenTelemetry observability instrumentation for vision-agents library.

This module defines metrics for the vision-agents library. It does NOT configure
OpenTelemetry providers - that is the responsibility of applications using this library.

Metrics are recorded by the MetricsCollector class, which subscribes to events from
LLM, STT, TTS, and turn detection components. This decouples instrumentation from
core logic and makes metrics collection opt-in.

For applications using this library:
    To enable telemetry, configure OpenTelemetry in your application before importing
    vision-agents components:

    ```python
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    # Setup Prometheus exporter
    reader = PrometheusMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    # Create your agent
    agent = Agent(llm=OpenAILLM(), stt=DeepgramSTT(), ...)

    # Opt-in to metrics collection
    from vision_agents.core.observability import MetricsCollector
    collector = MetricsCollector(agent)
    ```

    If no providers are configured, metrics will be no-ops.
"""

from opentelemetry import metrics

# Get meter using the library name
# Will use whatever provider the application has configured
# If no provider is configured, metrics will be no-ops
meter = metrics.get_meter("vision_agents.core")

# =============================================================================
# STT Metrics
# =============================================================================
stt_latency_ms = meter.create_histogram(
    "stt.latency.ms",
    unit="ms",
    description="STT processing latency",
)
stt_audio_duration_ms = meter.create_histogram(
    "stt.audio_duration.ms",
    unit="ms",
    description="Duration of audio processed by STT",
)
stt_errors = meter.create_counter(
    "stt.errors",
    description="STT errors",
)

# =============================================================================
# TTS Metrics
# =============================================================================
tts_latency_ms = meter.create_histogram(
    "tts.latency.ms",
    unit="ms",
    description="TTS synthesis latency",
)
tts_audio_duration_ms = meter.create_histogram(
    "tts.audio_duration.ms",
    unit="ms",
    description="Duration of synthesized audio",
)
tts_characters = meter.create_counter(
    "tts.characters",
    description="Characters synthesized by TTS",
)
tts_errors = meter.create_counter(
    "tts.errors",
    description="TTS errors",
)

# =============================================================================
# LLM Metrics
# =============================================================================
llm_latency_ms = meter.create_histogram(
    "llm.latency.ms",
    unit="ms",
    description="LLM response latency (request to complete response)",
)
llm_time_to_first_token_ms = meter.create_histogram(
    "llm.time_to_first_token.ms",
    unit="ms",
    description="LLM time to first token (streaming)",
)
llm_input_tokens = meter.create_counter(
    "llm.tokens.input",
    description="LLM input/prompt tokens consumed",
)
llm_output_tokens = meter.create_counter(
    "llm.tokens.output",
    description="LLM output/completion tokens generated",
)
llm_errors = meter.create_counter(
    "llm.errors",
    description="LLM errors",
)
llm_tool_calls = meter.create_counter(
    "llm.tool_calls",
    description="LLM tool/function calls executed",
)
llm_tool_latency_ms = meter.create_histogram(
    "llm.tool_latency.ms",
    unit="ms",
    description="LLM tool execution latency",
)

# =============================================================================
# Turn Detection Metrics
# =============================================================================
turn_duration_ms = meter.create_histogram(
    "turn.duration.ms",
    unit="ms",
    description="Duration of detected turns",
)
turn_trailing_silence_ms = meter.create_histogram(
    "turn.trailing_silence.ms",
    unit="ms",
    description="Trailing silence duration before turn end",
)

# =============================================================================
# Realtime LLM Metrics
# =============================================================================
realtime_sessions = meter.create_counter(
    "realtime.sessions",
    description="Realtime LLM sessions started",
)
realtime_session_duration_ms = meter.create_histogram(
    "realtime.session_duration.ms",
    unit="ms",
    description="Duration of realtime sessions",
)
realtime_audio_input_bytes = meter.create_counter(
    "realtime.audio.input.bytes",
    unit="By",
    description="Audio bytes sent to realtime LLM",
)
realtime_audio_output_bytes = meter.create_counter(
    "realtime.audio.output.bytes",
    unit="By",
    description="Audio bytes received from realtime LLM",
)
realtime_audio_input_duration_ms = meter.create_counter(
    "realtime.audio.input.duration.ms",
    unit="ms",
    description="Audio duration sent to realtime LLM",
)
realtime_audio_output_duration_ms = meter.create_counter(
    "realtime.audio.output.duration.ms",
    unit="ms",
    description="Audio duration received from realtime LLM",
)
realtime_responses = meter.create_counter(
    "realtime.responses",
    description="Realtime LLM responses received",
)
realtime_user_transcriptions = meter.create_counter(
    "realtime.transcriptions.user",
    description="User speech transcriptions from realtime LLM",
)
realtime_agent_transcriptions = meter.create_counter(
    "realtime.transcriptions.agent",
    description="Agent speech transcriptions from realtime LLM",
)
realtime_errors = meter.create_counter(
    "realtime.errors",
    description="Realtime LLM errors",
)

# =============================================================================
# VLM / Vision Metrics
# =============================================================================
vlm_inference_latency_ms = meter.create_histogram(
    "vlm.inference.latency.ms",
    unit="ms",
    description="VLM inference latency",
)
vlm_inferences = meter.create_counter(
    "vlm.inferences",
    description="VLM inference requests",
)
vlm_input_tokens = meter.create_counter(
    "vlm.tokens.input",
    description="VLM input tokens (text + image)",
)
vlm_output_tokens = meter.create_counter(
    "vlm.tokens.output",
    description="VLM output tokens",
)
vlm_errors = meter.create_counter(
    "vlm.errors",
    description="VLM errors",
)

# =============================================================================
# Video Processor Metrics
# =============================================================================
video_frames_processed = meter.create_counter(
    "video.frames.processed",
    description="Video frames processed",
)
video_processing_latency_ms = meter.create_histogram(
    "video.processing.latency.ms",
    unit="ms",
    description="Video frame processing latency",
)
video_detections = meter.create_counter(
    "video.detections",
    description="Objects/items detected in video",
)
