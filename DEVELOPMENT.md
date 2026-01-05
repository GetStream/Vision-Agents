## Overview

## To install:

In the project root, run:

```bash
uv venv --python 3.12.11
uv sync --all-extras --dev
pre-commit install
```

To setup your .env
```bash
cp env.example .env
```

## Running
```bash
uv run examples/01_simple_agent_example/simple_agent_example.py
```

### Running with a video file as input
```bash
uv run <path-to-example> --video-track-override <path-to-video>
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
rm -rf dist
git tag v0.0.15
uv run hatch version # this should show the right version
git push origin main --tags
uv build --all
uv publish
```

Common issues. If you have local changes (or ran build before you had the tag) you'll get this error

```
  Caused by: Upload failed with status code 400 Bad Request. Server says: 400 The use of local versions in <Version('0.0.16.dev0+gc7563254f.d20251008')> is not allowed. See https://packaging.python.org/specifications/core-metadata for more information.
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

**Audio**

* The agent listens to AudioReceivedEvent and calls simple_audio_response
* asyncio.create_task(self.llm.simple_audio_response(pcm_data))
* The STS writes on agent.llm.audio_track

**Video**

* The agent receives the video track, and calls agent.llm.watch_video_track
* The LLM uses the VideoForwarder to write the video to a websocket or webrtc connection
* The STS writes the reply on agent.llm.audio_track and the RealtimeTranscriptEvent / RealtimePartialTranscriptEvent

## Audio management

Some important things about audio inside the library:

1. WebRTC uses Opus 48khz stereo but inside the library audio is always in PCM format
2. Plugins / AI models work with different PCM formats, passing bytes around without a container type leads to kaos and is forbidden
3. PCM data is always passed around using the `PcmData` object which contains information about sample rate, channels and format
4. Audio resampling can be done using `PcmData.resample` method
5. Adjusting from stereo to mono and vice-versa can be done using the `PcmData.resample` method
6. `PcmData` comes with convenience constructor methods to build from bytes, iterators, ndarray, ...

## Simple Example

```python
import asyncio
from getstream.video.rtc.track_util import PcmData
from openai import AsyncOpenAI

async def example():
    client = AsyncOpenAI(api_key="sk-42")

    resp = await client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input="pcm is cool, give me some of that please",
        response_format="pcm",
    )

    # load response into PcmData, note that you need to specify sample_rate, channels and format
    pcm_data = PcmData.from_bytes(
        resp.content, sample_rate=24_000, channels=1, format="s16"
    )

    # check if pcm_data is stereo (it's not in this case ofc)
    print(pcm_data.stereo)

    # write the pcm to file
    with open("test.wav", "wb") as f:
        f.write(pcm_data.to_wav_bytes())

    # resample pcm to be 48khz stereo
    resampled_pcm = pcm_data.resample(48_000, 2)

    # play-out pcm using ffplay (you need to have PLAY_AUDIO=1 envvar)
    from vision_agents.core.tts.manual_test import play_pcm_with_ffplay

    await play_pcm_with_ffplay(resampled_pcm)

if __name__ == "__main__":
    asyncio.run(example())
```

### Testing audio manually

Sometimes you need to test audio manually, here's some tips:

1. Do not use earplugs when testing PCM playback ;)
2. You can use the `PcmData.to_wav_bytes` method to convert PCM into wav bytes (see `manual_tts_to_wav` for an example)
3. If you have `ffplay` installed, you can playback pcm directly to check if audio is correct

## Creating PcmData

### from_bytes
Build from raw PCM bytes

```python
from getstream.video.rtc import PcmData

PcmData.from_bytes(audio_bytes, sample_rate=16000, format=AudioFormat.S16, channels=1)
```

### from_numpy
Build from numpy arrays with automatic dtype/shape conversion

```python
# Automatically converts dtype and handles channel reshaping
PcmData.from_numpy(np.array([1, 2], np.int16), sample_rate=16000, format=AudioFormat.S16, channels=1)
```

### from_response
Construct from API response (bytes, iterators, async iterators, objects with .data)

```python
# Handles streaming responses
PcmData.from_response(
    response, sample_rate=16000, channels=1, format=AudioFormat.S16
)
```

### from_av_frame
Create from PyAV AudioFrame

```python
PcmData.from_av_frame(frame)
```

## Converting Format

### to_float32
Convert samples to float32 in [-1, 1]

```python
pcm_f32 = pcm.to_float32()
```

### to_int16
Convert samples to int16 PCM format

```python
pcm_s16 = pcm.to_int16()
```

### to_bytes
Return interleaved PCM bytes

```python
audio_bytes = pcm.to_bytes()
```

### to_wav_bytes
Return WAV file bytes (header + frames)

```python
wav_bytes = pcm.to_wav_bytes()
```

## Resampling and Channels

### resample

Resample to target sample rate and/or channels

```python
pcm = pcm.resample(16000, target_channels=1)  # to 16khz, mono
```

## Manipulating Audio

### append
Append another PcmData in-place (adjusts format/rate automatically)

```python
pcm.append(other_pcm)
```

### copy
Create a deep copy

```python
pcm_copy = pcm.copy()
```

### clear
Clear all samples in-place (keeps metadata)

```python
pcm.clear()
```

## Slicing and Chunking

### head
Keep only the first N seconds

```python
pcm_head = pcm.head(duration_s=3.0)
```

### tail
Keep only the last N seconds

```python
pcm_tail = pcm.tail(duration_s=5.0)
```

### chunks
Iterate over fixed-size chunks with optional overlap

```python
for chunk in pcm.chunks(chunk_size=4000, overlap=200):
    process(chunk)
```

# Audio Queue

Use `AudioQueue` from `utils.audio_queue` if you need to enqueue PCM and dequeue audio with a given format and duration.

```python
from vision_agents.core.utils.audio_queue import AudioQueue

queue = AudioQueue(buffer_limit_ms=1000)

# enqueue PcmData into the queue
queue.put(pcm)

# dequeue the oldest PcmData entry from the queue
await queue.get()

# dequeue 100ms of audio
pcm = await queue.get_duration(100)
```

# AudioTrack

Use `getstream.video.rtc.AudioTrack` if you need to publish audio using PyAV, this class ensures that `recv` paces audio correctly every 20ms.

- Use `.write()` method to enqueue audio (PcmData)
- Use `.flush()` to empty all the enqueued audio (eg. barge-in event)

By default AudioTrack holds 30s of audio in the buffer.

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
Keep typing simple. Use the `getstream.video.rtc.track_util.PcmData` type instead of bytes when passing around audio.
This prevents mistakes related to handling audio with different formats, sample rates etc.

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

### Profiling

The `Profiler` class uses `pyinstrument` to profile your agent's performance and generate an HTML report showing where time is spent during execution.

#### Example usage:

```python
from uuid import uuid4
from vision_agents.core import User, Agent
from vision_agents.core.profiling import Profiler
from vision_agents.plugins import getstream, gemini, deepgram, elevenlabs, vogent

async def start_agent() -> None:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My AI friend", id="agent"),
        instructions="You're a helpful assistant.",
        llm=gemini.LLM("gemini-2.0-flash"),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        turn_detection=vogent.TurnDetection(),
        profiler=Profiler(output_path='./profile.html'),  # Optional: specify output path
    )

    call = agent.edge.client.video.call("default", str(uuid4()))
    with await agent.join(call):
        await agent.simple_response("Hello!")
        await agent.finish()
```

The profiler automatically:
- Starts profiling when the agent is created
- Stops profiling when the agent finishes (on `AgentFinishEvent`)
- Saves an HTML report to the specified output path (default: `./profile.html`)

You can open the generated HTML file in a browser to view the performance profile, which shows a timeline of function calls and where time is spent during agent execution.


### Queuing

- Video: There is no reason to publish old video. So you want to cap the queue to x latest frames
- Audio: Writing faster than 1x causes audio glitches. So we need a queue.
- Audio: Writing slower than 1x also causes glitches. You need to write 0 frames
- Audio generated by LLM: The LLM -> TTS can generate a lot of audio. This has to be stopped when interrupt happens
- Gemini & Google generate at what pace?

### Tasks & Async

- Short running tasks should check if the connection is closed before doing work
- Long running tasks are should be cancelled when calling agent.close()
- Examples can be run with --debug to enable blockbuster and debug mode for async

### Video Frames & Tracks

- Track.recv errors will fail silently. The API is to return a frame. Never return None. and wait till the next frame is available
- When using frame.to_ndarray(format="rgb24") specify the format. Typically you want rgb24 when connecting/sending to Yolo etc
- QueuedVideoTrack is a writable/queued video track implementation which is useful when forwarding video


### Loading Resources in Plugins (aka "warmup")
Some plugins require to download and use external resources like models to work.  

For example:

- `TurnDetection` plugins using a Silero VAD model to detect voice activity in the audio track.
- Video processors using `YOLO` models

In order to standardise how these resources are loaded and to make it performant, the framework provides a special ABC
`vision_agents.core.warmup.Warmable`.  

To use it, simply subclass it and define the required methods.  
Note that `Warmable` supports generics to leverage type checking.

**Example:**

```python
from typing import Optional

from faster_whisper import WhisperModel

from vision_agents.core.stt import STT
from vision_agents.core.warmup import Warmable


# Add `Warmable[WhisperModel]` to the list of base classes.
# Here `WhisperModel` is the type returned by `on_warmup()` and cached by the global warmup cache.
class FasterWhisperSTT(STT, Warmable[WhisperModel]):
    """
    Faster-Whisper Speech-to-Text implementation.
    """

    def __init__(self):
        super().__init__()
        self._whisper_model: Optional[WhisperModel] = None

    async def on_warmup(self) -> WhisperModel:
        # Initialize the model here and return it
        # This method will be called once when the application starts.
        # The `whisper` object will be shared between all instances of `FasterWhisperSTT` in this app.
        whisper = WhisperModel("tiny")
        return whisper

    def on_warmed_up(self, whisper: WhisperModel) -> None:
        # Receive the warmed up instance and store it to the object.
        # This method will be called every time a new agent is initialized.
        # The warmup process is now complete.
        self._whisper_model = whisper
    
    ...
```



## Onboarding Plan for new contributors

**Audio Formats**

You'll notice that audio comes in many formats. PCM, wav, mp3. 16khz, 48khz.
Encoded as i16 or f32. Note that webrtc by default is 48khz.

A good first intro to audio formats can be found here:

**Using Cursor**

You can ask cursor something like "read @ai-plugin and build me a plugin called fish"
See the docs folder for other ai instruction files

**Learning Roadmap**

1. Quick refresher on audio formats
2. Build a TTS integration
3. Build a STT integration
4. Build an LLM integration
5. Write a pytest test with a fixture
