# Video/Vision Agents by Stream

Video/Vision AI agents running on [Stream's edge network](https://getstream.io/video/).
Open Agent library. Goal is to support our video/audio competitors.

-  **Video AI**: Built for real-time video AI. Combine Yolo, Roboflow and others with realtime models
-  **Low Latency**: Join quickly (500ms) and low audio/video latency (30ms)
-  **Open**: Built by Stream, but use any video edge network that you like
-  **Native APIs**: Native SDK methods from OpenAI (create response), Gemini (generate) and Claude (create message). So you're never behind on the latest features
-  **SDKs**: SDKs for React, Android, iOS, Flutter, React, React Native and Unity.

## Examples

### Golf Coaching Example

It's easy to build a golf coaching AI. The example below combines Yolo with openAI realtime.

Demo video

```python
# partial example, full example: examples/03_golf_coach_example/golf_coach_example.py
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="Help users perfect their swing. Read @golf_coach.md",
    # openai realtime, no need to set tts, or sst (though that's also supported)
    llm=openai.Realtime(model="gpt-realtime"),
    processors=[
        YOLOPoseProcessor()
    ],  # processors can fetch extra data, check images/audio data or transform video
)
```

### Cluely style example

Cluely offers realtime coaching via an invisible overlay. This example shows you how you can build your own app like Cluely.
It combines Gemini realtime (to watch your screen), and doesn't broadcast audio (only text).

Demo video

```python
# partial example, full example: examples/...
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="You are silently helping the user pass this interview. See @interview_coach.md",
    # gemini realtime, no need to set tts, or sst (though that's also supported)
    llm=gemini.Realtime()
)
```

### Dota Coaching Example

Video agents typically need a lot of context to be effective. 
This example combines OpenDota, Stratz, valve game state together with Gemini to improve your Dota skills.
It highlights how you can combine multiple APIs for a full coaching experience.

Demo video

```python
# partial example, full example: examples/...
agent = Agent(
    edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
    agent_user=agent_user,  # the user object for the agent (name, image etc)
    instructions="You are silently helping the user pass this interview. See @interview_coach.md",
    # gemini realtime, no need to set tts, or sst (though that's also supported)
    llm=gemini.Realtime()
)
```

### Other examples

* Video AI avatar
* React UI customization
* Linear standup agent

## Docs

TODO Link to docs

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

**Stream agents**

This python framework gives you full control.

## 📦 Installation

```bash
# Install dependencies using uv
uv add openai python-dotenv stream_agents
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

### Queuing

- Video: There is no reason to publish old video. So you want to cap the queue to x latest frames
- Audio: Writing faster than 1x causes audio glitches. So we need a queue.
- Audio: Writing slower than 1x also causes glitches. You need to write 0 frames
- Audio generated by LLM: The LLM -> TTS can generate a lot of audio. This has to be stopped when interrupt happens
- Gemini & Google generate at what pace?

### Other

- aiortc will automatically resample to 48kHZ when publishing/receiving. so that's why a track at 16 or 24khz works

## Inspiration

- Livekit Agents: Great syntax, Livekit only
- Pipecat: Flexible, but more verbose. Open, we will add support for Stream
- OpenAI Agents: Focused on openAI only, but we will try to add support

## Stream Agents or Proxy

The proxy mode which handles the openAI/Stream connection is a good option if you don't need to run any additional AI models.
If all you need is low latency integration between stream and openAI, that's a good option.
It's available for JS & Python.
