# otel_setup.py
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Point these at your collector (default shown)
OTLP_ENDPOINT = "http://localhost:4317"

resource = Resource.create(
    {
        "service.name": "voice-agent",
        "service.version": "1.0.0",
    }
)

# --- Traces ---
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT))
)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# --- Metrics ---
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTLP_ENDPOINT)
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)


meter = metrics.get_meter("voice-agent.latency")

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
tts_first_byte_ms = meter.create_histogram(
    "tts.first_byte.ms", unit="ms", description="TTS time to first audio byte"
)
tts_bytes_streamed = meter.create_counter(
    "tts.bytes.streamed", unit="By", description="Bytes sent/received for TTS"
)
tts_errors = meter.create_counter("tts.errors", description="TTS errors")

inflight_ops = meter.create_up_down_counter(
    "voice.ops.inflight", description="Inflight voice ops"
)

CALL_ATTRS = {
    "provider": "deepgram",  # or "whisper", "revai", "gcloud", etc.
    "model": "nova-2",  # your model id
    "lang": "en-US",  # BCP-47 / ISO code
    "transport": "http",  # or "websocket", "grpc"
    "streaming": True,  # True/False
}

with tracer.start_as_current_span("stt.request", kind=trace.SpanKind.CLIENT) as span:
    pass

span = tracer.start_span("stt.request")
span.end()
