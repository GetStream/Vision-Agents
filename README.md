<img width="1280" height="360" alt="Readme" src="assets/repo_image.png" />

# Open Vision Agents by Stream

[![build](https://github.com/GetStream/Vision-Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/GetStream/Vision-Agents/actions)
[![PyPI version](https://badge.fury.io/py/vision-agents.svg)](http://badge.fury.io/py/vision-agents)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vision-agents.svg)
[![License](https://img.shields.io/github/license/GetStream/Vision-Agents)](https://github.com/GetStream/Vision-Agents/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1108586339550638090)](https://discord.gg/RkhX9PxMS6)

---

## Build Real-Time Vision AI Agents

https://github.com/user-attachments/assets/d9778ab9-938d-4101-8605-ff879c29b0e4

### Multi-modal AI agents that watch, listen, and understand video.

Vision Agents give you the building blocks to create intelligent, low-latency video experiences powered by your models,
your infrastructure, and your use cases.

### Key Highlights

- **Video AI:** Built for real-time video AI. Combine YOLO, Roboflow, and others with Gemini/OpenAI in real-time.
- **Low Latency:** Join quickly (500ms) and maintain audio/video latency under 30ms
  using [Stream's edge network](https://getstream.io/video/).
- **Open:** Built by Stream, but works with any video edge network.
- **Native APIs:** Native SDK methods from OpenAI (`create response`), Gemini (`generate`), and Claude (
  `create message`) — always access the latest LLM capabilities.
- **SDKs:** SDKs for React, Android, iOS, Flutter, React Native, and Unity, powered by Stream's ultra-low-latency
  network.

https://github.com/user-attachments/assets/d66587ea-7af4-40c4-9966-5c04fbcf467c

---

## See It In Action

### Sports Coaching

https://github.com/user-attachments/assets/d1258ac2-ca98-4019-80e4-41ec5530117e

This example shows you how to build golf coaching AI with YOLO and Gemini Live.
Combining a fast object detection model (like YOLO) with a full realtime AI is useful for many different video AI use
cases.
For example: Drone fire detection, sports/video game coaching, physical therapy, workout coaching, just dance style
games etc.

```python
# partial example, full example: examples/02_golf_coach_example/golf_coach_example.py
agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    instructions="Read @golf_coach.md",
    llm=gemini.Realtime(fps=10),
    # llm=openai.Realtime(fps=1), # Careful with FPS can get expensive
    processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt", device="cuda")],
)
```

### Security Camera with Package Theft Detection

https://github.com/user-attachments/assets/92a2cdd8-909c-46d8-aab7-039a90efc186

This example shows a security camera system that detects faces, tracks packages and detects when a package is stolen. It
automatically generates "WANTED" posters, posting them to X in real-time.

It combines face recognition, YOLOv11 object detection, Nano Banana and Gemini for a complete security workflow with
voice interaction.

```python
# partial example, full example: examples/04_security_camera_example/security_camera_example.py
security_processor = SecurityCameraProcessor(
    fps=5,
    model_path="weights_custom.pt",  # YOLOv11 for package detection
    package_conf_threshold=0.7,
)

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Security AI", id="agent"),
    instructions="Read @instructions.md",
    processors=[security_processor],
    llm=gemini.LLM("gemini-2.5-flash-lite"),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
)
```

### Cluely style Invisible Assistant (coming soon)

Apps like Cluely offer realtime coaching via an invisible overlay. This example shows you how you can build your own
invisible assistant.
It combines Gemini realtime (to watch your screen and audio), and doesn't broadcast audio (only text). This approach
is quite versatile and can be used for: Sales coaching, job interview cheating, physical world/ on the job coaching with
glasses

Demo video

```python
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="You are silently helping the user pass this interview. See @interview_coach.md",
    # gemini realtime, no need to set tts, or sst (though that's also supported)
    llm=gemini.Realtime()
)
```

## Quick Start

**Step 1: Install via uv**

`uv add vision-agents`

**Step 2: (Optional) Install with extra integrations**

`uv add "vision-agents[getstream, openai, elevenlabs, deepgram]"`

**Step 3: Obtain your Stream API credentials**

Get a free API key from [Stream](https://getstream.io/). Developers receive **333,000 participant minutes** per month,
plus extra credits via the Maker Program.

## Features

| **Feature**                         | **Description**                                                                                                                                       |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **True real-time via WebRTC**       | Stream directly to model providers that support it for instant visual understanding.                                                                  |
| **Interval/processor pipeline**     | For providers without WebRTC, process frames with pluggable video processors (e.g., YOLO, Roboflow, or custom PyTorch/ONNX) before/after model calls. |
| **Turn detection & diarization**    | Keep conversations natural; know when the agent should speak or stay quiet and who's talking.                                                         |
| **Voice activity detection (VAD)**  | Trigger actions intelligently and use resources efficiently.                                                                                          |
| **Speech↔Text↔Speech**              | Enable low-latency loops for smooth, conversational voice UX.                                                                                         |
| **Tool/function calling**           | Execute arbitrary code and APIs mid-conversation. Create Linear issues, query weather, trigger telephony, or hit internal services.                   |
| **Built-in memory via Stream Chat** | Agents recall context naturally across turns and sessions.                                                                                            |
| **Text back-channel**               | Message the agent silently during a call.                                                                                                             |
| **Phone and RAG**                   | Interact with the Agent via inbound or outbound phone calls using Twilio and Turbopuffer                                                              |

## Out-of-the-Box Integrations

| **Plugin Name** | **Description**                                                                                                                                                                                                                         | **Docs Link**                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| AWS Bedrock     | Realtime speech-to-speech plugin using Amazon Nova models with automatic reconnection                                                                                                                                                   | [AWS](https://visionagents.ai/integrations/aws-bedrock)                                          |
| AWS Polly       | TTS plugin using Amazon's cloud-based service with natural-sounding voices and neural engine support                                                                                                                                    | [AWS Polly](https://visionagents.ai/integrations/aws-polly)                                      |
| Cartesia        | TTS plugin for realistic voice synthesis in real-time voice applications                                                                                                                                                                | [Cartesia](https://visionagents.ai/integrations/cartesia)                                        |
| Decart          | Real-time AI video transformation service for applying artistic styles and effects to video streams                                                                                                                                     | [Decart](https://visionagents.ai/integrations/decart)                                            |
| Deepgram        | STT plugin for fast, accurate real-time transcription with speaker diarization                                                                                                                                                          | [Deepgram](https://visionagents.ai/integrations/deepgram)                                        |
| ElevenLabs      | TTS plugin with highly realistic and expressive voices for conversational agents                                                                                                                                                        | [ElevenLabs](https://visionagents.ai/integrations/elevenlabs)                                    |
| Fast-Whisper    | High-performance STT plugin using OpenAI's Whisper model with CTranslate2 for fast inference                                                                                                                                            | [Fast-Whisper](https://visionagents.ai/integrations/fast-whisper)                                |
| Fish Audio      | STT and TTS plugin with automatic language detection and voice cloning capabilities                                                                                                                                                     | [Fish Audio](https://visionagents.ai/integrations/fish)                                          |
| Gemini          | Realtime API for building conversational agents with support for both voice and video                                                                                                                                                   | [Gemini](https://visionagents.ai/integrations/gemini)                                            |
| HeyGen          | Realtime interactive avatars powered by [HeyGen](https://heygen.com/)                                                                                                                                                                   | [HeyGen](https://visionagents.ai/integrations/heygen)                                            |
| Inworld         | TTS plugin with high-quality streaming voices for real-time conversational AI agents                                                                                                                                                    | [Inworld](https://visionagents.ai/integrations/inworld)                                          |
| Kokoro          | Local TTS engine for offline voice synthesis with low latency                                                                                                                                                                           | [Kokoro](https://visionagents.ai/integrations/kokoro)                                            |
| Moondream       | Moondream provides realtime detection and VLM capabilities. Developers can choose from using the hosted API or running locally on their CUDA devices. Vision Agents supports Moondream's Detect, Caption and VQA skills out-of-the-box. | [Moondream](https://visionagents.ai/integrations/moondream)                                      |
| NVIDIA Cosmos 2 | VLM plugin using NVIDIA's Cosmos 2 models for video understanding with automatic frame buffering and streaming responses                                                                                                                | [NVIDIA](https://visionagents.ai/integrations/nvidia)                                            |
| OpenAI          | Realtime API for building conversational agents with out of the box support for real-time video directly over WebRTC, LLMs and Open AI TTS                                                                                              | [OpenAI](https://visionagents.ai/integrations/openai)                                            |
| OpenRouter      | LLM plugin providing access to multiple providers (Anthropic, Google, OpenAI) through a unified API                                                                                                                                     | [OpenRouter](https://visionagents.ai/integrations/openrouter)                                    |
| Qwen            | Realtime audio plugin using Alibaba's Qwen3 with native audio output and built-in speech recognition                                                                                                                                    | [Qwen](https://visionagents.ai/integrations/qwen)                                                |
| Roboflow        | Object detection processor using Roboflow's hosted API or local RF-DETR models                                                                                                                                                          | [Roboflow](https://visionagents.ai/integrations/roboflow)                                        |
| Smart Turn      | Advanced turn detection system combining Silero VAD, Whisper, and neural models for natural conversation flow                                                                                                                           | [Smart Turn](https://visionagents.ai/integrations/smart-turn)                                    |
| TurboPuffer     | RAG plugin using TurboPuffer for hybrid search (vector + BM25) with Gemini embeddings for retrieval augmented generation                                                                                                                | [TurboPuffer](https://visionagents.ai/guides/rag)                                                |
| Twilio          | Voice call integration plugin enabling bidirectional audio streaming via Twilio Media Streams with call registry and audio conversion                                                                                                   | [Twilio](https://github.com/GetStream/Vision-Agents/tree/main/examples/03_phone_and_rag_example) |
| Ultralytics     | Real-time pose detection processor using YOLO models with skeleton overlays                                                                                                                                                             | [Ultralytics](https://visionagents.ai/integrations/ultralytics)                                  |
| Vogent          | Neural turn detection system for intelligent turn-taking in voice conversations                                                                                                                                                         | [Vogent](https://visionagents.ai/integrations/vogent)                                            |
| Wizper          | STT plugin with real-time translation capabilities powered by Whisper v3                                                                                                                                                                | [Wizper](https://visionagents.ai/integrations/wizper)                                            |
| xAI             | LLM plugin using xAI's Grok models with advanced reasoning and real-time knowledge                                                                                                                                                      | [xAI](https://visionagents.ai/integrations/xai)                                                  |

## Processors

Processors let your agent **manage state** and **handle audio/video** in real-time.

They take care of the hard stuff, like:

- Running smaller models
- Making API calls
- Transforming media

… so you can focus on your agent logic.

## Documentation

Check out our getting started guide at [VisionAgents.ai](https://visionagents.ai/).

- **Quickstart:** [Building a Voice AI app](https://visionagents.ai/introduction/voice-agents)
- **Quickstart:** [Building a Video AI app](https://visionagents.ai/introduction/video-agents)
- **Tutorial:** [Building a real-time meeting assistant](https://github.com/GetStream/Vision-Agents/tree/main/examples/01_simple_agent_example)
- **Tutorial:** [Building real-time sports coaching](https://github.com/GetStream/Vision-Agents/tree/main/examples/02_golf_coach_example)

---

## External WebRTC Integration Guide

Vision Agents supports **external WebRTC providers** in addition to Stream's edge network. This allows you to integrate local devices, custom RTC infrastructure, or embedded systems directly into your Vision AI agents.

### Architecture Overview

```
┌─────────────────┐          ┌──────────────────┐          ┌─────────────────┐
│   Local Edge    │          │   EdgeTransport  │          │   LLM Provider  │
│  (Your Device)  │  ◄──────►│   (localrtc)     │  ◄──────►│  (Gemini/GPT)   │
│                 │          │                  │          │                 │
│  • Microphone   │   Audio  │  • Format        │  Audio   │  • Realtime API │
│  • Camera       │   Video  │    Negotiation   │  Video   │  • STT/TTS      │
│  • Speakers     │          │  • GStreamer     │  Text    │  • Vision       │
└─────────────────┘          └──────────────────┘          └─────────────────┘
```

**Data Flow:**
1. **Local Edge** captures audio/video from your devices (microphone, camera)
2. **EdgeTransport** negotiates audio formats and manages media streams
3. **LLM Provider** receives streams, processes with AI, and sends responses back
4. **Local Edge** plays audio responses through speakers

### Quick Start

**Install the localrtc plugin:**

```bash
uv add vision-agents-plugin-localrtc
```

**Basic example:**

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import localrtc, gemini

# Create local edge transport
edge = localrtc.Edge(
    audio_device="default",    # System default microphone
    speaker_device="default",  # System default speakers
    video_device=0,            # First camera
    sample_rate=16000,         # 16kHz input (output auto-negotiated)
    channels=1,                # Mono audio
)

# Create agent with local devices
agent = Agent(
    edge=edge,
    agent_user=User(name="Local AI", id="agent"),
    instructions="You're a helpful voice AI assistant.",
    llm=gemini.Realtime(),  # Output format auto-negotiated to 24kHz mono
)

# Join and run
call = await agent.create_call("default", "my-call")
async with agent.join(call):
    await agent.simple_response("Say hello!")
    await agent.finish()
```

### Configuration Reference

#### Environment Variables

All configuration values can be set via environment variables or programmatically:

| Variable | Default | Description |
|----------|---------|-------------|
| **Audio Input** |
| `VA_AUDIO_INPUT_SAMPLE_RATE` | `16000` | Sample rate for microphone capture (Hz) |
| `VA_AUDIO_INPUT_CHANNELS` | `1` | Number of input channels (1=mono, 2=stereo) |
| `VA_AUDIO_INPUT_BUFFER_DURATION` | `2.0` | Input buffer duration (seconds) |
| `VA_AUDIO_CAPTURE_CHUNK_DURATION` | `0.1` | Audio capture chunk size (seconds) |
| **Audio Output** |
| `VA_AUDIO_OUTPUT_SAMPLE_RATE` | `24000` | Default output sample rate (Hz, auto-negotiated) |
| `VA_AUDIO_OUTPUT_CHANNELS` | `1` | Default output channels (auto-negotiated) |
| `VA_AUDIO_OUTPUT_BUFFER_SIZE_MS` | `10000` | Output buffer size (milliseconds) |
| `VA_AUDIO_PLAYBACK_CHUNK_DURATION` | `0.05` | Audio playback chunk size (seconds) |
| **Audio Timing** |
| `VA_AUDIO_LOOP_SLEEP_INTERVAL` | `0.001` | Loop sleep interval (seconds) |
| `VA_AUDIO_FLUSH_POLL_INTERVAL` | `0.01` | Flush polling interval (seconds) |
| `VA_AUDIO_ERROR_RETRY_DELAY` | `0.1` | Error retry delay (seconds) |
| `VA_AUDIO_THREAD_JOIN_TIMEOUT` | `2.0` | Thread join timeout (seconds) |
| `VA_AUDIO_EOS_WAIT_TIME` | `0.1` | End-of-stream wait time (seconds) |
| **Video** |
| `VA_VIDEO_DEFAULT_WIDTH` | `640` | Default video width (pixels) |
| `VA_VIDEO_DEFAULT_HEIGHT` | `480` | Default video height (pixels) |
| `VA_VIDEO_DEFAULT_FPS` | `30` | Default video frame rate (fps) |
| `VA_VIDEO_FORMAT` | `"BGR"` | Video color format |
| `VA_VIDEO_MAX_BUFFERS` | `1` | Maximum video buffer count |
| **GStreamer** |
| `VA_AUDIO_BIT_DEPTH` | `16` | Audio bit depth (16 or 24) |
| `VA_GSTREAMER_APPSINK_NAME` | `"sink"` | GStreamer appsink element name |
| `VA_GSTREAMER_APPSRC_NAME` | `"src"` | GStreamer appsrc element name |
| `VA_GSTREAMER_AUDIO_LAYOUT` | `"interleaved"` | Audio channel layout |

#### Programmatic Configuration

```python
# Using LocalEdge directly
from vision_agents.plugins.localrtc import Edge, LocalEdgeConfig, AudioConfig, VideoConfig

# Method 1: Use environment variables (recommended for production)
edge = localrtc.Edge(
    audio_device="default",
    speaker_device="default",
    video_device=0,
)

# Method 2: Explicit configuration
config = LocalEdgeConfig(
    audio=AudioConfig(
        input_sample_rate=16000,
        output_sample_rate=24000,
        input_channels=1,
        output_channels=1,
    ),
    video=VideoConfig(
        default_width=1280,
        default_height=720,
        default_fps=30,
    ),
)
edge = localrtc.Edge(config=config)

# Method 3: Override defaults programmatically
edge = localrtc.Edge(
    sample_rate=16000,  # Input sample rate
    channels=1,          # Input channels
    # Output format is auto-negotiated with LLM provider
)
```

### Audio Format Negotiation

Vision Agents automatically negotiates audio formats between your local device and the LLM provider to ensure compatibility.

**How it works:**

1. You configure **input** format (microphone capture):
   ```python
   edge = localrtc.Edge(
       sample_rate=16000,  # 16kHz input
       channels=1,         # Mono input
   )
   ```

2. During `agent.join()`, the system queries the LLM's audio requirements:
   ```python
   # Gemini Realtime requires 24kHz mono
   gemini.Realtime().get_audio_requirements()
   # Returns: AudioFormat(sample_rate=24000, channels=1)
   ```

3. Output format is automatically configured to match the LLM:
   - **Gemini Realtime**: 24kHz mono
   - **GetStream**: 48kHz stereo
   - **Custom providers**: Queries `llm.get_audio_requirements()`

4. GStreamer handles resampling automatically:
   ```
   Input (16kHz mono) → GStreamer audioresample → Output (24kHz mono) → Gemini
   ```

**Provider-specific formats:**

| Provider | Input (Recommended) | Output (Auto) | Notes |
|----------|---------------------|---------------|-------|
| Gemini Realtime | 16kHz mono | 24kHz mono | Native Gemini format |
| OpenAI Realtime | 16kHz mono | 24kHz mono | Compatible with Gemini |
| GetStream Edge | 16kHz mono | 48kHz stereo | High-quality WebRTC |
| Custom LLM | 16kHz mono | Via `get_audio_requirements()` | Implement method |

### Custom GStreamer Pipelines

For advanced use cases like Raspberry Pi, embedded systems, or custom hardware, you can specify custom GStreamer pipelines:

```python
# Raspberry Pi with ALSA and V4L2
custom_pipeline = {
    "audio_source": "alsasrc device=hw:1,0 ! audioconvert ! audioresample",
    "video_source": "v4l2src device=/dev/video0 ! videoconvert",
    "audio_sink": "alsasink device=hw:0,0",
}

edge = localrtc.Edge(
    sample_rate=16000,
    channels=1,
    custom_pipeline=custom_pipeline,
)
```

**GStreamer pipeline elements:**

- **`alsasrc`**: ALSA audio input (Linux)
- **`alsasink`**: ALSA audio output (Linux)
- **`v4l2src`**: Video4Linux2 camera (Linux)
- **`audioconvert`**: Audio format conversion
- **`audioresample`**: Sample rate conversion
- **`videoconvert`**: Video format conversion
- **`appsrc`/`appsink`**: Application source/sink (automatically added)

See [examples/localrtc/raspberry_pi_gstreamer.py](examples/localrtc/raspberry_pi_gstreamer.py) for a complete Raspberry Pi example.

### Device Discovery

**List available devices:**

```python
from vision_agents.plugins import localrtc

devices = localrtc.Edge.list_devices()

# Audio inputs (microphones)
for device in devices["audio_inputs"]:
    print(f"{device['index']}: {device['name']}")
# Output: 0: Built-in Microphone
#         1: USB Audio Device

# Audio outputs (speakers)
for device in devices["audio_outputs"]:
    print(f"{device['index']}: {device['name']}")
# Output: 0: Built-in Speakers
#         1: HDMI Audio

# Video inputs (cameras)
for device in devices["video_inputs"]:
    print(f"{device['index']}: {device['name']}")
# Output: 0: FaceTime HD Camera
#         1: USB Webcam
```

**Use specific device:**

```python
# By index (integer)
edge = localrtc.Edge(
    audio_device=1,       # Second microphone
    speaker_device=0,     # First speaker
    video_device=1,       # Second camera
)

# By name (string)
edge = localrtc.Edge(
    audio_device="USB Audio Device",
    speaker_device="Built-in Speakers",
    video_device=0,
)

# Use "default" for system default
edge = localrtc.Edge(
    audio_device="default",
    speaker_device="default",
)
```

### Troubleshooting Guide

#### Common GStreamer Errors

**Error: "GStreamer is not available"**
```bash
# Ubuntu/Debian
sudo apt-get install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-good

# macOS
brew install gstreamer gst-plugins-good gst-plugins-bad pygobject3

# Verify installation
gst-inspect-1.0 --version
```

**Error: "No such device" (ALSA)**
```bash
# List ALSA devices
arecord -L  # Input devices
aplay -L    # Output devices

# Test microphone
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav

# Test speaker
aplay -D hw:0,0 test.wav
```

**Error: "Cannot open video device"**
```bash
# List video devices
v4l2-ctl --list-devices

# Check device permissions
ls -l /dev/video0
sudo usermod -a -G video $USER  # Add user to video group

# Test camera
gst-launch-1.0 v4l2src device=/dev/video0 ! autovideosink
```

#### Audio Format Mismatches

**Symptom: Distorted or choppy audio**

Check for format mismatch warnings in logs:
```
WARNING: Audio format mismatch detected
  LLM requires: 24000 Hz, 1 channel
  Edge configured: 16000 Hz, 1 channel
  Resampling will be applied
```

**Solution:**
```python
# Ensure input format is correctly set
edge = localrtc.Edge(
    sample_rate=16000,  # Match your microphone
    channels=1,         # Use mono for voice
)
# Output format is automatically negotiated - no action needed
```

#### High CPU Usage

**Symptom: Agent consumes excessive CPU**

**Solutions:**
1. Lower video frame rate:
   ```python
   agent = Agent(
       edge=edge,
       llm=gemini.Realtime(fps=1),  # 1 frame/second instead of 10
   )
   ```

2. Disable video entirely:
   ```python
   edge = localrtc.Edge(
       audio_device="default",
       speaker_device="default",
       # video_device=None,  # No video
   )
   ```

3. Use hardware acceleration (Raspberry Pi):
   ```python
   custom_pipeline = {
       "video_source": "v4l2src device=/dev/video0 ! omxh264enc ! videoconvert",
   }
   ```

#### No Audio Output

**Symptom: Agent doesn't speak**

1. Check audio format negotiation:
   ```python
   # Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Verify speaker device:
   ```python
   devices = localrtc.Edge.list_devices()
   print("Available speakers:", devices["audio_outputs"])
   ```

3. Test speaker directly:
   ```bash
   # Generate test tone
   gst-launch-1.0 audiotestsrc ! audioconvert ! autoaudiosink
   ```

4. Check volume and mute:
   ```bash
   # Linux
   alsamixer

   # macOS
   # Use System Preferences > Sound
   ```

### Production Deployment Checklist

#### Security

- [ ] **API Keys**: Store LLM API keys in environment variables, not in code
  ```bash
  export GOOGLE_API_KEY="your-key-here"
  export OPENAI_API_KEY="your-key-here"
  ```

- [ ] **Device Permissions**: Request user permission before accessing microphone/camera
  ```python
  # Inform users which devices will be accessed
  print("This app will access:")
  print("- Microphone for voice input")
  print("- Camera for video (optional)")
  print("- Speakers for audio output")
  ```

- [ ] **Data Privacy**: Document what data is sent to LLM providers
  - Audio/video streams sent to Gemini/OpenAI
  - No data stored locally by default
  - Check LLM provider's data retention policies

- [ ] **Network Security**: Use HTTPS for all API communications
  - Gemini/OpenAI APIs use HTTPS by default
  - Verify SSL certificates are validated

#### Performance

- [ ] **Audio Configuration**: Optimize for voice use cases
  ```python
  edge = localrtc.Edge(
      sample_rate=16000,  # 16kHz is sufficient for voice
      channels=1,         # Mono reduces bandwidth by 50%
  )
  ```

- [ ] **Video Frame Rate**: Set appropriate FPS for your use case
  ```python
  # Real-time coaching: 3-10 fps
  llm = gemini.Realtime(fps=5)

  # Security camera: 1-3 fps
  llm = gemini.Realtime(fps=1)

  # Voice-only: no video
  llm = gemini.Realtime(fps=0)
  ```

- [ ] **Buffer Sizes**: Tune for latency vs reliability
  ```bash
  # Low latency (50ms)
  export VA_AUDIO_OUTPUT_BUFFER_SIZE_MS=50

  # Balanced (100ms, default)
  export VA_AUDIO_OUTPUT_BUFFER_SIZE_MS=100

  # High reliability (500ms)
  export VA_AUDIO_OUTPUT_BUFFER_SIZE_MS=500
  ```

- [ ] **Resource Limits**: Monitor CPU/memory usage
  ```python
  import psutil

  # Log resource usage
  cpu_percent = psutil.cpu_percent(interval=1)
  memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
  print(f"CPU: {cpu_percent}%, Memory: {memory_mb:.1f} MB")
  ```

#### Monitoring

- [ ] **Logging**: Enable appropriate log levels
  ```python
  import logging

  # Production: INFO level
  logging.basicConfig(level=logging.INFO)

  # Debug issues: DEBUG level
  logging.basicConfig(level=logging.DEBUG)
  ```

- [ ] **Error Handling**: Implement graceful error recovery
  ```python
  try:
      async with agent.join(call):
          await agent.finish()
  except Exception as e:
      logger.error(f"Agent error: {e}")
      # Notify user, restart agent, etc.
  ```

- [ ] **Health Checks**: Monitor agent status
  ```python
  # Check if agent is still responsive
  async def health_check():
      try:
          await agent.send_text("ping")
          return True
      except:
          return False
  ```

- [ ] **Metrics Collection**: Track usage and performance
  - Audio/video stream duration
  - LLM response latency
  - Error rates and types
  - Device availability

#### Testing

- [ ] **Device Compatibility**: Test on target hardware
  - Test all microphone/speaker/camera combinations
  - Verify GStreamer pipeline compatibility
  - Check OS-specific device paths

- [ ] **Format Negotiation**: Verify with your LLM provider
  ```python
  # Log negotiated format
  edge = localrtc.Edge(sample_rate=16000, channels=1)
  agent = Agent(edge=edge, llm=your_llm)

  # Check logs for:
  # "Negotiated output format: 24000 Hz, 1 channel"
  ```

- [ ] **Network Conditions**: Test with various connectivity
  - High latency networks (>100ms)
  - Low bandwidth (<1 Mbps)
  - Intermittent connectivity

- [ ] **Load Testing**: Verify performance under load
  - Multiple concurrent agents
  - Long-running sessions (>1 hour)
  - Rapid agent creation/destruction

#### Deployment

- [ ] **Dependencies**: Document all requirements
  ```txt
  # requirements.txt
  vision-agents-stream>=0.3
  vision-agents-plugin-localrtc>=0.3
  vision-agents-plugin-gemini>=0.3  # or your LLM plugin
  python-dotenv
  ```

- [ ] **Environment Setup**: Create deployment guide
  ```bash
  # Install system dependencies
  sudo apt-get install gstreamer1.0-tools python3-gi

  # Install Python packages
  pip install -r requirements.txt

  # Configure environment
  cp .env.example .env
  # Edit .env with your API keys
  ```

- [ ] **Process Management**: Use a process supervisor
  ```bash
  # systemd (Linux)
  sudo systemctl enable my-agent.service
  sudo systemctl start my-agent.service

  # pm2 (Node.js)
  pm2 start agent.py --interpreter python3
  ```

- [ ] **Auto-restart**: Handle crashes and restarts
  ```python
  # Automatic retry on failure
  MAX_RETRIES = 3
  for attempt in range(MAX_RETRIES):
      try:
          await run_agent()
          break
      except Exception as e:
          logger.error(f"Attempt {attempt+1} failed: {e}")
          if attempt < MAX_RETRIES - 1:
              await asyncio.sleep(5)  # Wait before retry
  ```

### Example Applications

**Basic agent** - Minimal setup with default devices:
[examples/localrtc/basic_agent.py](examples/localrtc/basic_agent.py)

**Multi-component agent** - Separate LLM, STT, and TTS:
[examples/localrtc/multi_component_agent.py](examples/localrtc/multi_component_agent.py)

**Raspberry Pi with GStreamer** - Custom pipelines for embedded systems:
[examples/localrtc/raspberry_pi_gstreamer.py](examples/localrtc/raspberry_pi_gstreamer.py)

---

## Examples

| 🔮 Demo Applications                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                                                                         |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| <br><h3>Cartesia</h3>Using Cartesia's Sonic 3 model to visually look at what's in the frame and tell a story with emotion.<br><br>• Real-time visual understanding<br>• Emotional storytelling<br>• Frame-by-frame analysis<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/plugins/cartesia/example)                                                                                                                                                    | <img src="assets/demo_gifs/cartesia.gif" width="320" alt="Cartesia Demo">               |
| <br><h3>Realtime Stable Diffusion</h3>Realtime stable diffusion using Vision Agents and Decart's Mirage 2 model to create interactive scenes and stories.<br><br>• Real-time video restyling<br>• Interactive scene generation<br>• Stable diffusion integration<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/plugins/decart/example)                                                                                                                 | <img src="assets/demo_gifs/mirage.gif" width="320" alt="Mirage Demo">                   |
| <br><h3>Golf Coach</h3>Using Gemini Live together with Vision Agents and Ultralytics YOLO, we're able to track the user's pose and provide realtime actionable feedback on their golf game.<br><br>• Real-time pose tracking<br>• Actionable coaching feedback<br>• YOLO pose detection<br>• Gemini Live integration<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/examples/02_golf_coach_example)                                                     | <img src="assets/demo_gifs/golf.gif" width="320" alt="Golf Coach Demo">                 |
| <br><h3>GeoGuesser</h3>Together with OpenAI Realtime and Vision Agents, we can take GeoGuesser to the next level by asking it to identify places in our real world surroundings.<br><br>• Real-world location identification<br>• OpenAI Realtime integration<br>• Visual scene understanding<br><br> [>Source Code and tutorial](https://visionagents.ai/integrations/openai#openai-realtime)                                                                                                    | <img src="assets/demo_gifs/geoguesser.gif" width="320" alt="GeoGuesser Demo">           |
| <br><h3>Phone and RAG</h3>Interact with your Agent over the phone using Twilio. This example demonstrates how to use TurboPuffer for Retrieval Augmented Generation (RAG) to give your agent specialized knowledge.<br><br>• Inbound/Outbound telephony<br>• Twilio Media Streams integration<br>• Vector search with TurboPuffer<br>• Retrieval Augmented Generation<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/examples/03_phone_and_rag_example) | <img src="assets/demo_gifs/va_phone.png" width="320" alt="Phone and RAG Demo">          |
| <br><h3>Security Camera</h3>A security camera with face recognition, package detection and automated theft response. Generates WANTED posters with Nano Banana and posts them to X when packages disappear.<br><br>• Face detection & named recognition<br>• YOLOv11 package detection<br>• Automated WANTED poster generation<br>• Real-time X posting<br><br> [>Source Code and tutorial](https://github.com/GetStream/Vision-Agents/tree/main/examples/04_security_camera_example)             | <img src="assets/demo_gifs/security_camera.gif" width="320" alt="Security Camera Demo"> |

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md)

## Open Platform

Want to add your platform or provider? Reach out to **nash@getstream.io**.

## Awesome Video AI

Our favorite people & projects to follow for vision AI

| [<img src="https://github.com/user-attachments/assets/9149e871-cfe8-4169-a4ce-4073417e645c" width="80"/>](https://x.com/demishassabis) | [<img src="https://github.com/user-attachments/assets/2e1335d3-58af-4988-b879-1db8d862cd34" width="80"/>](https://x.com/OfficialLoganK) | [<img src="https://github.com/user-attachments/assets/c9249ae9-e66a-4a70-9393-f6fe4ab5c0b0" width="80"/>](https://x.com/ultralytics) |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|
|                 [@demishassabis](https://x.com/demishassabis)<br>CEO @ Google DeepMind<br><sub>Won a Nobel prize</sub>                 |           [@OfficialLoganK](https://x.com/OfficialLoganK)<br>Product Lead @ Gemini<br><sub>Posts about robotics vision</sub>            |       [@ultralytics](https://x.com/ultralytics)<br>Various fast vision AI models<br><sub>Pose, detect, segment, classify</sub>       |

| [<img src="https://github.com/user-attachments/assets/c1fe873d-6f41-4155-9be1-afc287ca9ac7" width="80"/>](https://x.com/skalskip92) | [<img src="https://github.com/user-attachments/assets/43359165-c23d-4d5d-a5a6-1de58d71fabd" width="80"/>](https://x.com/moondreamai) | [<img src="https://github.com/user-attachments/assets/490d349c-7152-4dfb-b705-04e57bb0a4ca" width="80"/>](https://x.com/kwindla) |
|:-----------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
|          [@skalskip92](https://x.com/skalskip92)<br>Open Source Lead @ Roboflow<br><sub>Building tools for vision AI</sub>          |       [@moondreamai](https://x.com/moondreamai)<br>The tiny vision model that could<br><sub>Lightweight, fast, efficient</sub>       |                [@kwindla](https://x.com/kwindla)<br>Pipecat / Daily<br><sub>Sharing AI and vision insights</sub>                 |

| [<img src="https://github.com/user-attachments/assets/d7ade584-781f-4dac-95b8-1acc6db4a7c4" width="80"/>](https://x.com/juberti) | [<img src="https://github.com/user-attachments/assets/00a1ed37-3620-426d-b47d-07dd59c19b28" width="80"/>](https://x.com/romainhuet) | [<img src="https://github.com/user-attachments/assets/eb5928c7-83b9-4aaa-854f-1d4f641426f2" width="80"/>](https://x.com/thorwebdev) |
|:--------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
|               [@juberti](https://x.com/juberti)<br>Head of Realtime AI @ OpenAI<br><sub>Realtime AI systems</sub>                |                [@romainhuet](https://x.com/romainhuet)<br>Head of DX @ OpenAI<br><sub>Developer tooling & APIs</sub>                |                    [@thorwebdev](https://x.com/thorwebdev)<br>Eleven Labs<br><sub>Voice and AI experiments</sub>                    |

| [<img src="https://github.com/user-attachments/assets/ab5ef918-7c97-4c6d-be10-2e2aeefec015" width="80"/>](https://x.com/mervenoyann) | [<img src="https://github.com/user-attachments/assets/af936e13-22cf-4000-a35b-bfe30d44c320" width="80"/>](https://x.com/stash_pomichter) |         [<img src="https://pbs.twimg.com/profile_images/1893061651152121856/Op4W8mza_400x400.jpg" width="80"/>](https://x.com/Mentraglass)          |
|:------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
|               [@mervenoyann](https://x.com/mervenoyann)<br>Hugging Face<br><sub>Posts extensively about Video AI</sub>               |          [@stash_pomichter](https://x.com/stash_pomichter)<br>Spatial memory for robots<br><sub>Robotics & AI navigation</sub>           | [@Mentraglass](https://x.com/Mentraglass)<br>Open-source smart glasses<br><sub>Open-Source, hackable AR glasses with AI capabilities built in</sub> |

| [<img src="https://pbs.twimg.com/profile_images/1664559115581145088/UMD1vtMw_400x400.jpg" width="80"/>](https://x.com/vikhyatk) |
|:-------------------------------------------------------------------------------------------------------------------------------:|
|        [@vikhyatk](https://x.com/vikhyatk)<br>AI Engineer<br><sub>Open-source AI projects, Creator of Moondream AI</sub>        |

## Inspiration

- Livekit Agents: Great syntax, Livekit only
- Pipecat: Flexible, but more verbose.
- OpenAI Agents: Focused on openAI only

## Roadmap

### 0.1 – First Release - Oct

- Working TTS, Gemini & OpenAI

### 0.2 - Simplification - Nov

- Simplified the library & improved code quality
- Deepgram Nova 3, Elevenlabs Scribe 2, Fish, Moondream, QWen3, Smart turn, Vogent, Inworld, Heygen, AWS and more
- Improved openAI & Gemini realtime performance
- Audio & Video utilities

### 0.3 - Examples and Deploys - Jan

- Production-grade HTTP API for agent deployment (`uv run <agent.py> serve`)
- Metrics & Observability stack
- Phone/voice integration with RAG capabilities
- 10 new LLM
  plugins ([AWS Nova 2](plugins/aws), [Qwen 3 Realtime](plugins/qwen), [NVIDIA Cosmos 2](plugins/nvidia), [Pocket TTS](plugins/pocket), [Deepgram TTS](plugins/deepgram), [OpenRouter](plugins/openrouter), [HuggingFace Inference](plugins/huggingface), [Roboflow](plugins/roboflow), [Twilio](plugins/twilio), [Turbopuffer](plugins/turbopuffer))
- Real-world
  examples ([security camera](examples/05_security_camera_example), [phone integration](examples/03_phone_and_rag_example), [football commentator](examples/04_football_commentator_example), [Docker deployment with GPU support](examples/07_deploy_example), [agent server](examples/08_agent_server_example))
- Stability: Fixes for participant sync, video frame handling, agent lifecycle, and screen sharing

### 0.4 Documentation/polish

- Excellence on documentation/polish
- Better Roboflow annotation docs
- Automated workflows for maintenance
- Local camera/audio support AND/OR WebRTC connection
- Embedded/robotics examples

## Vision AI limitations

Video AI is the frontier of AI. The state of the art is changing daily to help models understand live video.
While building the integrations, here are the limitations we've noticed (Dec 2025)

* Video AI struggles with small text. If you want the AI to read the score in a game it will often get it wrong and
  hallucinate
* Longer videos can cause the AI to lose context. For instance if it's watching a soccer match it will get confused
  after 30 seconds
* Most applications require a combination of small specialized models like Yolo/Roboflow/Moondream, API calls to get
  more context and larger models like gemini/openAI
* Image size & FPS need to stay relatively low due to performance constraints
* Video doesn’t trigger responses in realtime models. You always need to send audio/text to trigger a response.

## We are hiring

Join the team behind this project - we’re hiring a Staff Python Engineer to architect, build, and maintain a powerful
toolkit for developers integrating voice and video AI into their products.

[Apply here](https://jobs.ashbyhq.com/stream/3bea7dba-54e1-4c71-aa02-712a075842df?utm_source=Jmv9QOkznl)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GetStream/vision-agents&type=timeline&legend=top-left)](https://www.star-history.com/#GetStream/vision-agents&type=timeline&legend=top-left)
