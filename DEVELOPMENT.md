## Overview

## To install:

In the project root, run:

```bash
uv venv --python 3.12.11
uv sync --all-extras --dev
```

To setup your .env
```bash
cp env.example .env
```

## Running
```bash
uv run examples/01_simple_agent_example/simple_agent_example.py
```

## Tests

Everything other than integration tests

```
uv run py.test -m "not integration" -n auto
```

Integration test. (requires secrets in place, see .env setup)
```
uv run py.test -m "integration" -n auto
```

Plugin tests (TODO: not quite right. uv env is different for each plugin)

```
uv run py.test plugins/*/tests/*.py -m "not integration"
```

### Check

Shortcut to ruff, mypy and non integration tests:

```
uv run python dev.py check
```

### Formatting

```
uv run ruff check --fix
```

### Mypy type checks


```
uv run mypy --install-types --non-interactive -p vision_agents
```

```
uv run mypy --install-types --non-interactive --exclude 'plugins/.*/tests/.*' plugins
```

## Release

Create a new release on Github, CI handles the rest. If you do need to do it manually follow these instructions:

```
uv build --all
```

## Architecture

To see how the agent work open up agents.py

### STT & TTS flow

* The agent listens to AudioReceivedEvent and forwards that to STT.
* STT then fires the STTPartialTranscriptEvent and STTTranscriptEvent event. 
* The agent receives this event and calls agent.llm.simple_response.
* The LLM triggers LLMResponseEvent, and the agent calls 
* await self.tts.send(llm_response.text)

### Realtime STS flow

** Audio **

* The agent listens to AudioReceivedEvent and calls simple_audio_response
* asyncio.create_task(self.llm.simple_audio_response(pcm_data))
* The STS writes on agent.llm.audio_track

** Video **

* The agent receives the video track, and calls agent.llm._watch_video_track
* The LLM uses the VideoForwarder to write the video to a websocket or webrtc connection
* The STS writes the reply on agent.llm.audio_track and the RealtimeTranscriptEvent / RealtimePartialTranscriptEvent

## Dev / Contributor Guidelines

### Light wrapping

AI is changing daily. This makes it important to use light wrapping. IE

```python
tts = ElevenLabsTTS(client=ElevenLabs())
```

Note how the ElevenLabsTTS handles standardization.
But if the init for ElevenLabs changes, nothing breaks.
If features are added to the client, you can use them easily via tts.client

### Typing

Avoid using Union types or complicated composite types.
Keep typing simple. Use the PcmAudio type instead of bytes when passing around audio.
This prevents mistakes related to different audio formats.

### Testing

Many of the underlying APIs change daily. To ensure things work we keep 2 sets of tests. Integration tests and unit tests.
Integration tests run once a day to verify that changes to underlying APIs didn't break the framework. Some testing guidelines

- Every plugin needs an integration test
- Limit usage of response capturing style testing. (since they diverge from reality)

### Observability

- Traces and metrics go to Prometheus and OpenTelemetry
- Metrics on performance of TTS, STT, LLM, Turn detection and connection to realtime edge.
- Integration with external LLM observability solutions

#### Example setup for tracing and Jaeger:

**Step 1 - Install open telemetry OTLP exporter**

```bash
# with uv:
uv install opentelemetry-sdk opentelemetry-exporter-otlp

# or with pip:
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
`````

**Step 2 - Setup tracing instrumentation in your code**

Make sure to setup the instrumentation before you start the agent/server

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create(
    {
        "service.name": "agents",
    }
)
tp = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)

tp.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(tp)
```

**Step 3 - Run Jaeger**

```bash
docker run --rm -it \
         -e COLLECTOR_OTLP_ENABLED=true \
         -p 16686:16686 -p 4317:4317 -p 4318:4318 \
         jaegertracing/all-in-one:1.51```
```

After this, you can run your code and see the traces in Jaeger at `http://localhost:16686`

#### Example setup for metrics with Prometheus:

**Step 1 - Install prometheus exporter**

```bash
# with uv:
uv install opentelemetry-exporter-prometheus prometheus-client

# or with pip:
pip install opentelemetry-exporter-prometheus prometheus-client
```

**Step 2 - Setup metrics instrumentation in your code**

Make sure to setup the instrumentation before you start the agent/server

```python
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server

resource = Resource.create(
    {
        "service.name": "my-service-name",
    }
)

reader = PrometheusMetricReader()
metrics.set_meter_provider(
    MeterProvider(resource=resource, metric_readers=[reader])
)

start_http_server(port=9464)
```

You can now see the metrics at `http://localhost:9464/metrics` (make sure that your Python program keeps running), after this you can setup your Prometheus server to scrape this endpoint.


### Queuing

- Video: There is no reason to publish old video. So you want to cap the queue to x latest frames
- Audio: Writing faster than 1x causes audio glitches. So we need a queue.
- Audio: Writing slower than 1x also causes glitches. You need to write 0 frames
- Audio generated by LLM: The LLM -> TTS can generate a lot of audio. This has to be stopped when interrupt happens
- Gemini & Google generate at what pace?

### Tasks

- Short running tasks should check if the connection is closed before doing work
- Long running tasks are should be cancelled when calling agent.close()

### Video Frames & Tracks

- Track.recv errors will fail silently. The API is to return a frame. Never return None. and wait till the next frame is available
- When using frame.to_ndarray(format="rgb24") specify the format. Typically you want rgb24 when connecting/sending to Yolo etc
