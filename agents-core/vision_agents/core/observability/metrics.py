"""OpenTelemetry observability instrumentation for vision-agents library.

This module defines metrics and tracers for the vision-agents library. It does NOT
configure OpenTelemetry providers - that is the responsibility of applications using
this library.

For applications using this library:
    To enable telemetry, configure OpenTelemetry in your application before importing
    vision-agents components:

    ```python
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    # Configure your service
    resource = Resource.create({
        "service.name": "my-voice-app",
        "service.version": "1.0.0",
    })

    # Setup trace provider
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
    )
    trace.set_tracer_provider(trace_provider)

    # Setup metrics provider
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="http://localhost:4317")
    )
    metrics_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(metrics_provider)

    # Now import and use vision-agents
    from vision_agents.core.tts import TTS
    ```

    If no providers are configured, metrics and traces will be no-ops.
"""

from opentelemetry import trace, metrics

# Get tracer and meter using the library name
# These will use whatever providers the application has configured
# If no providers are configured, they will be no-ops
tracer = trace.get_tracer("vision_agents.core")
meter = metrics.get_meter("vision_agents.core")

stt_latency_ms = meter.create_histogram(
    "stt.latency.ms", unit="ms", description="Total STT latency"
)
stt_first_byte_ms = meter.create_histogram(
    "stt.first_byte.ms", unit="ms", description="STT time to first token/byte"
)
stt_bytes_streamed = meter.create_counter(
    "stt.bytes.streamed", unit="By", description="Bytes received from STT"
)
stt_errors = meter.create_counter("stt.errors", description="STT errors")

tts_latency_ms = meter.create_histogram(
    "tts.latency.ms", unit="ms", description="Total TTS latency"
)
tts_errors = meter.create_counter("tts.errors", description="TTS errors")
tts_events_emitted = meter.create_counter(
    "tts.events.emitted", description="Number of TTS events emitted"
)

inflight_ops = meter.create_up_down_counter(
    "voice.ops.inflight", description="Inflight voice ops"
)
