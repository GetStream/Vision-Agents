# Video/Vision Agents by Stream

Video/Vision AI agents on [Stream's edge network](https://getstream.io/video/).

-  **Video AI**: Built for real-time video AI. Combine Yolo, Roboflow and others with realtime models
-  **Low Latency**: Quickly join (500ms) and fast audio latency (30ms)
-  **Open**: Built by Stream, but use any video edge network that you like
-  **Voice & Chat**: Store & query conversation history
-  **Native APIs**: Native SDK methods from OpenAI (create response), Gemini (generate) and Claude (create message). So you're never behind on the latest features

Open Agent library. Goal is to support most of our video/audio competitors. (See adding support section)
SDKs for React, Android, iOS, Flutter, React, React Native and Unity.

## Examples

### Cluely example

Listen to voice and watch the screen but don't respond. Show suggestions as text/chat.

Demo video

```python
# partial example, see the example file for full details
```

### Golf Coaching Example

Use Yolo to determine body position. Share this with gemini for live coaching.

Demo video

```python
# partial example, see the example file for full details
```

### Dota Coaching Example

Use API calls to retrieve game state, while also analysing the gameplay.
OpenAI + API calls.

Demo video

### Security Camera Example

Demo video

## OpenAI Proxy mode vs Stream agents

** OpenAI Proxy mode **:

For use cases that only require you to connect openAI & Stream you can consider
https://getstream.io/video/voice-agents/
This is a direct proxy written in Go and maintained by Stream.
You can use it with any programming language/ SDK.

Benefits:

* Hosted
* Faster/lowest latency
* All programming languages

** Stream agents **

This python framework gives you full control.

## 📦 Installation

```bash
# Install dependencies using uv
uv add openai python-dotenv stream_agents

# Or with pip
pip install openai python-dotenv stream_agents
```

## ⚡ Quick Start - Cluely style AI

```python
from agents import Agent

# Roboflow for object detection (finetuned)
# load docs from @ai-dota-coaching.md
# use speech to speech (STS) gemini model
agent = Agent(
    pre_processors=[Roboflow(), dota_api("gameid")],
    interval=1
second,
llm = GeminiSTS(),
# turn_detection=your_turn_detector
)

# Join a Stream video call
await agent.join(call)

# gemini is available at
agent.llm.client

# history at
agent._conversation
```




## ⚡ Quick Start - Video AI Coach

```python
from agents import Agent

# Roboflow for object detection (finetuned)
# load docs from @ai-dota-coaching.md
# use speech to speech (STS) gemini model
agent = Agent(
    pre_processors=[Roboflow(), dota_api("gameid")],
    interval=1 second,
    llm=GeminiSTS(),
    # turn_detection=your_turn_detector
)

# Join a Stream video call
await agent.join(call)
```

## Video Transform

Roboflow server side transform of video


```python
from agents import Agent


# Create an agent with the exact syntax you requested
agent = Agent(
    video_transformer=RoboflowTransform() # transform live video from python (instead of on-device). use for AI avatars
)

# Join a Stream video call
await agent.join(call)
```

## Quick Example - Voice AI (TTS, STT) and STS

```python
from agents import Agent
from models import OpenAILLM

# Create an AI model
llm = OpenAILLM(
    api_key="<your-api-key>",
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello there"},
    ],
    default_temperature=0.8,
)

# Create an agent with the exact syntax you requested
agent = Agent(
    stt=your_stt_service,
    pre_processors=[Roboflow()],
    llm=llm,
    tts=your_tts_service,
    turn_detection=your_turn_detector
)

sts_model = OpenAIRealtime(
    api_key="<your-api-key>",
    model="gpt-4o-realtime-preview",
    voice="Pluck"
)
speechToSpeechAgent = Agent(
    sts=sts_model,
)

# Join a Stream video call
await agent.join(call)
```


## 📚 Examples

Other example to build
- Simple image
- Simple video capture
- Simple audio capture
- Moderation example (AWS image. comarketing)
- SIP with twilio
- Cluely clone

Check out the [`examples/`](./examples/) directory for complete working examples:

- **`examples/main.py`** - Basic TTS bot with ElevenLabs
- **`examples/example_agent.py`** - Agent framework with tools and pre-processors
- **`examples/example_openai_model.py`** - OpenAI model integration
- **`examples/example_agent_with_openai.py`** - Complete AI agent with OpenAI

```bash
# Run any example
python examples/example_agent_with_openai.py
```

## 🔧 Configuration

Create a `.env` file with your API keys:

```bash
# Stream API (required)
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
EXAMPLE_BASE_URL=https://pronto-staging.getstream.io

# ElevenLabs TTS (for voice examples)
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# OpenAI (for AI model examples)
OPENAI_API_KEY=your_openai_api_key
```

## 🏗️ Architecture

### Core Components

- **`agents/`** - Agent framework and protocols
- **`models/`** - AI model implementations (OpenAI, extensible to others)
- **`examples/`** - Working examples and demos
- **`tests/`** - Comprehensive test suite

### Agent System

```python
agent = Agent(
    tools=[external_api_tool],           # External API integrations
    pre_processors=[data_processor],     # Input data processing
    llm=ai_model,                      # AI model for responses
    stt=speech_to_text,                  # Speech recognition
    tts=text_to_speech,                  # Voice synthesis
    turn_detection=turn_detector         # Conversation management
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_agent.py -v
pytest tests/test_models.py -v
```

## 🛣️ Roadmap


## Competitors & Partners

Reach out to nash@getstream.io, and we'll collaborate on getting you added

## 🤝 Dev Guidelines

### Light wrapping

AI is changing daily. This makes it important to use light wrapping. IE

```python
tts = ElevenLabsTTS(client=ElevenLabsClient())
```

Note how the ElevenLabsTTS handles standardization.
But if the init for ElevenLabsClient changes, nothing breaks.
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

## Inspiration

- Livekit Agents: Great syntax, Livekit only
- Pipecat: Flexible, but more verbose. Open, we will add support for Stream
- OpenAI Agents: Focused on openAI only, but we will try to add support

## Stream Agents or Proxy

The proxy mode which handles the openAI/Stream connection is a good option if you don't need to run any additional AI models.
If all you need is low latency integration between stream and openAI, that's a good option.
It's available for JS & Python.
