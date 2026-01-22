# Local RTC Examples

This directory contains production-ready examples demonstrating how to build Vision AI agents with the External WebRTC integration using the `localrtc` plugin.

## Overview

The **Local RTC** plugin (`vision-agents-plugin-localrtc`) enables Vision Agents to work with local audio/video devices (microphone, camera, speakers) without requiring Stream's edge network or other external RTC infrastructure. This is ideal for:

- Development and testing without cloud dependencies
- Privacy-focused applications that keep data local
- Desktop applications with direct device access
- Embedded systems (Raspberry Pi, IoT devices)
- Custom RTC infrastructure integration

## Examples

### 1. Basic Agent ([basic_agent.py](basic_agent.py))

**Minimal setup demonstrating best practices for External WebRTC integration.**

**Features:**
- Default device selection (microphone, camera, speakers)
- Gemini Realtime API for multimodal interaction
- Automatic audio format negotiation
- Production-ready error handling

**Best for:**
- Getting started with Local RTC
- Understanding the standard API patterns
- Voice/video AI prototypes

**Run:**
```bash
# Install dependencies
uv add vision-agents-plugin-localrtc vision-agents-plugin-gemini

# Set up environment
export GOOGLE_API_KEY="your-google-api-key"

# Run the agent
python basic_agent.py
```

**Key concepts demonstrated:**
- `localrtc.Edge()` - EdgeTransport for local devices
- `agent.create_call()` - Standard call creation pattern
- `async with agent.join(call)` - Automatic resource cleanup
- `agent.simple_response()` - Initial interaction
- Audio format negotiation (16kHz input → 24kHz output for Gemini)

---

### 2. Multi-Component Agent ([multi_component_agent.py](multi_component_agent.py))

**Demonstrates separating LLM, STT, and TTS into independent components.**

**Features:**
- OpenAI LLM for language processing
- Deepgram STT for speech recognition
- ElevenLabs TTS for voice synthesis
- Local RTC for device I/O
- Mix-and-match component architecture

**Best for:**
- Production applications requiring specific providers
- Cost optimization (e.g., cheaper STT + premium TTS)
- A/B testing different component combinations
- Multi-language support

**Run:**
```bash
# Install dependencies
uv add vision-agents-plugin-localrtc \
       vision-agents-plugin-openai \
       vision-agents-plugin-deepgram \
       vision-agents-plugin-elevenlabs

# Set up environment
export OPENAI_API_KEY="your-openai-key"
export DEEPGRAM_API_KEY="your-deepgram-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"

# Run the agent
python multi_component_agent.py
```

**Data flow:**
```
Microphone → STT (Deepgram) → Text → LLM (OpenAI) → Text → TTS (ElevenLabs) → Speakers
```

**Key concepts demonstrated:**
- Independent component configuration
- Component composition flexibility
- Provider-agnostic architecture
- Advanced agent construction

---

### 3. Raspberry Pi GStreamer ([raspberry_pi_gstreamer.py](raspberry_pi_gstreamer.py))

**Custom GStreamer pipelines for embedded systems and Raspberry Pi.**

**Features:**
- ALSA audio input/output (Linux audio system)
- V4L2 video capture (Video4Linux2)
- Custom GStreamer pipeline configuration
- Device path discovery instructions
- Performance optimization for embedded systems

**Best for:**
- Raspberry Pi projects
- Embedded Linux systems
- Custom hardware integration
- Direct device control
- IoT applications

**Run on Raspberry Pi:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-gi \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    alsa-utils \
    v4l-utils

# Find your devices
arecord -L          # List audio input devices
aplay -L            # List audio output devices
v4l2-ctl --list-devices  # List video devices

# Update device paths in raspberry_pi_gstreamer.py

# Install Python dependencies
uv add vision-agents-plugin-localrtc vision-agents-plugin-gemini

# Set up environment
export GOOGLE_API_KEY="your-google-api-key"

# Run the agent
python raspberry_pi_gstreamer.py
```

**Key concepts demonstrated:**
- Custom GStreamer pipeline configuration
- ALSA device paths (`hw:1,0`, `plughw:0,0`)
- V4L2 device paths (`/dev/video0`)
- Hardware-specific optimization
- Device discovery and testing

---

## Common Patterns

### Device Discovery

All examples support device enumeration before creating the agent:

```python
from vision_agents.plugins import localrtc

# List all available devices
devices = localrtc.Edge.list_devices()

print("Audio Inputs:")
for device in devices["audio_inputs"]:
    print(f"  {device['index']}: {device['name']}")

print("\nAudio Outputs:")
for device in devices["audio_outputs"]:
    print(f"  {device['index']}: {device['name']}")

print("\nVideo Inputs:")
for device in devices["video_inputs"]:
    print(f"  {device['index']}: {device['name']}")
```

### Device Selection

```python
# Method 1: Use device index (integer)
edge = localrtc.Edge(
    audio_device=1,       # Second microphone
    speaker_device=0,     # First speaker
    video_device=0,       # First camera
)

# Method 2: Use device name (string)
edge = localrtc.Edge(
    audio_device="USB Audio Device",
    speaker_device="Built-in Speakers",
    video_device=0,
)

# Method 3: Use "default" for system default
edge = localrtc.Edge(
    audio_device="default",
    speaker_device="default",
    video_device=0,
)
```

### Audio Format Configuration

```python
# Configure input format (microphone)
edge = localrtc.Edge(
    sample_rate=16000,  # 16kHz for voice input
    channels=1,         # Mono audio
)

# Output format is automatically negotiated based on LLM provider:
# - Gemini Realtime: 24kHz mono
# - GetStream: 48kHz stereo
# - Custom: queries llm.get_audio_requirements()
```

### Error Handling

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create agent with error handling
try:
    call = await agent.create_call("default", "my-call")
    async with agent.join(call):
        await agent.simple_response("Say hello!")
        await agent.finish()
except Exception as e:
    logger.error(f"Agent error: {e}")
    # Handle gracefully: restart, notify user, etc.
```

### Custom GStreamer Pipelines

```python
# Raspberry Pi example
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

# macOS example with AVFoundation
custom_pipeline = {
    "audio_source": "avfaudiosrc ! audioconvert ! audioresample",
    "video_source": "avfvideosrc ! videoconvert",
}

# Linux example with PulseAudio
custom_pipeline = {
    "audio_source": "pulsesrc ! audioconvert ! audioresample",
    "audio_sink": "pulsesink",
}
```

---

## Environment Configuration

Create a `.env` file in this directory with your API keys:

```bash
# Copy example
cp ../.env.example .env

# Edit with your keys
# Required for basic_agent.py and raspberry_pi_gstreamer.py
GOOGLE_API_KEY=your-google-api-key-here

# Required for multi_component_agent.py
OPENAI_API_KEY=your-openai-api-key-here
DEEPGRAM_API_KEY=your-deepgram-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
```

## Audio Configuration Environment Variables

Fine-tune audio behavior using environment variables:

```bash
# Input configuration
export VA_AUDIO_INPUT_SAMPLE_RATE=16000      # 16kHz for voice
export VA_AUDIO_INPUT_CHANNELS=1             # Mono
export VA_AUDIO_INPUT_BUFFER_DURATION=2.0    # 2 second buffer

# Output configuration (auto-negotiated, but can override)
export VA_AUDIO_OUTPUT_SAMPLE_RATE=24000     # 24kHz (Gemini native)
export VA_AUDIO_OUTPUT_CHANNELS=1            # Mono
export VA_AUDIO_OUTPUT_BUFFER_SIZE_MS=10000  # 10 second buffer

# Timing
export VA_AUDIO_CAPTURE_CHUNK_DURATION=0.1   # 100ms chunks
export VA_AUDIO_PLAYBACK_CHUNK_DURATION=0.05 # 50ms chunks

# Video configuration
export VA_VIDEO_DEFAULT_WIDTH=640
export VA_VIDEO_DEFAULT_HEIGHT=480
export VA_VIDEO_DEFAULT_FPS=30
```

See the main [README.md](../../README.md#configuration-reference) for the complete list.

---

## Troubleshooting

### No audio input detected

```bash
# Check microphone access
# macOS: System Preferences > Security & Privacy > Microphone
# Linux: Check device permissions
ls -l /dev/snd/*

# List available devices
python -c "
from vision_agents.plugins import localrtc
devices = localrtc.Edge.list_devices()
print('Audio inputs:', devices['audio_inputs'])
"

# Test microphone with GStreamer
gst-launch-1.0 autoaudiosrc ! audioconvert ! autoaudiosink
```

### No audio output

```bash
# Check speaker volume and mute status
# macOS: System Preferences > Sound
# Linux: alsamixer or pavucontrol

# List available output devices
python -c "
from vision_agents.plugins import localrtc
devices = localrtc.Edge.list_devices()
print('Audio outputs:', devices['audio_outputs'])
"

# Test speakers with GStreamer
gst-launch-1.0 audiotestsrc ! audioconvert ! autoaudiosink
```

### No video from camera

```bash
# Check camera permissions
# macOS: System Preferences > Security & Privacy > Camera
# Linux: Check /dev/video* permissions

# List cameras
python -c "
from vision_agents.plugins import localrtc
devices = localrtc.Edge.list_devices()
print('Video inputs:', devices['video_inputs'])
"

# Test camera with GStreamer
gst-launch-1.0 autovideosrc ! videoconvert ! autovideosink

# Linux: Test with v4l2
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all
```

### GStreamer not found

```bash
# Ubuntu/Debian
sudo apt-get install python3-gi gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-alsa

# macOS
brew install gstreamer gst-plugins-base gst-plugins-good \
    gst-plugins-bad pygobject3

# Verify installation
gst-inspect-1.0 --version
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst"
```

### High CPU usage

```python
# Lower video frame rate
agent = Agent(
    edge=edge,
    llm=gemini.Realtime(fps=1),  # 1 frame/second instead of 10
)

# Disable video entirely
agent = Agent(
    edge=edge,
    llm=gemini.Realtime(fps=0),  # No video
)

# Use lower video resolution
export VA_VIDEO_DEFAULT_WIDTH=320
export VA_VIDEO_DEFAULT_HEIGHT=240
```

### Audio format mismatch warnings

```
WARNING: Audio format mismatch detected
  LLM requires: 24000 Hz, 1 channel
  Edge configured: 48000 Hz, 2 channel
```

**Solution:**
This is informational - GStreamer automatically resamples. However, for best performance, configure input to match common voice settings:

```python
edge = localrtc.Edge(
    sample_rate=16000,  # Standard voice sample rate
    channels=1,         # Mono for voice
)
# Output format (24kHz for Gemini) is automatically negotiated
```

---

## Performance Optimization

### For Desktop Applications

```python
# Balanced settings for desktop
edge = localrtc.Edge(
    audio_device="default",
    speaker_device="default",
    video_device=0,
    sample_rate=16000,  # Voice quality
    channels=1,         # Mono
)

agent = Agent(
    edge=edge,
    llm=gemini.Realtime(fps=3),  # 3 frames/second
)
```

### For Raspberry Pi

```python
# Optimized for Pi 3/4
edge = localrtc.Edge(
    sample_rate=16000,
    channels=1,
    custom_pipeline={
        "audio_source": "alsasrc device=hw:1,0 ! audioconvert ! audioresample",
        "video_source": "v4l2src device=/dev/video0 ! videoconvert",
        "audio_sink": "alsasink device=hw:0,0",
    },
)

agent = Agent(
    edge=edge,
    llm=gemini.Realtime(fps=1),  # 1 frame/second
)
```

### For Voice-Only Applications

```python
# Maximum performance for voice
edge = localrtc.Edge(
    audio_device="default",
    speaker_device="default",
    # video_device=None,  # No video
    sample_rate=16000,
    channels=1,
)

agent = Agent(
    edge=edge,
    llm=gemini.Realtime(fps=0),  # Disable video
)
```

---

## Production Checklist

Before deploying your Local RTC agent to production:

- [ ] Test on target hardware (OS, devices, performance)
- [ ] Verify device permissions (microphone, camera)
- [ ] Configure environment variables (API keys, settings)
- [ ] Enable appropriate logging (INFO for prod, DEBUG for issues)
- [ ] Implement error handling and auto-restart
- [ ] Document required API keys and dependencies
- [ ] Test network resilience (latency, connectivity)
- [ ] Monitor resource usage (CPU, memory, bandwidth)
- [ ] Set up health checks and metrics
- [ ] Review LLM provider data policies
- [ ] Test audio format negotiation with your LLM
- [ ] Verify GStreamer pipeline compatibility

See the main [README.md](../../README.md#production-deployment-checklist) for the complete checklist.

---

## Additional Resources

- **Main Documentation**: [README.md](../../README.md#external-webrtc-integration-guide)
- **Configuration Reference**: [README.md](../../README.md#configuration-reference)
- **Troubleshooting Guide**: [README.md](../../README.md#troubleshooting-guide)
- **Vision Agents Docs**: [visionagents.ai](https://visionagents.ai/)
- **API Reference**: [visionagents.ai/api](https://visionagents.ai/api)

## Getting Help

- **GitHub Issues**: [github.com/GetStream/Vision-Agents/issues](https://github.com/GetStream/Vision-Agents/issues)
- **Discord Community**: [discord.gg/RkhX9PxMS6](https://discord.gg/RkhX9PxMS6)
- **Email Support**: nash@getstream.io

---

## License

See [LICENSE](../../LICENSE) in the repository root.
