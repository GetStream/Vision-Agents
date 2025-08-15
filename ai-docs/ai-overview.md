## AI Overview 
This project is an Agent framework built on Stream's Python AI library for Video and Voice calling. 

The source code for the Python library which contains the WebRTC and Call logic can be found here: https://github.com/GetStream/stream-py/tree/webrtc

## Project Structure
Most of the project code and logic lives in `/agents`. As an agentic framework on top of Stream's WebRTC SDK, this project 
aims to create a framework that's simple yet pwoerful for developers to use to build various types of applications using real-time 
voice and video agents. 

The core API looks something like this:

```python 
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

### Turn Detection

Turn detection is integrated via a small, unified interface that the `Agent` understands. Most providers (Fal.ai, Krisp, etc.) can be wrapped by our adapter so your agent code stays simple:

```python
from turn_detection import FalSmartTurnDetector
from agents import TurnDetectionAdapter

detector = FalSmartTurnDetector()
turn_detection = TurnDetectionAdapter(detector, agent_user_id=agent.bot_id)

agent = Agent(
    stt=your_stt_service,
    llm=llm,
    tts=your_tts_service,
    turn_detection=turn_detection,
)
```

Advanced detectors may also implement the interface directly. The Agent uses a small set of methods: `start/stop`, `add_participant`, `process_audio` or `process_audio_track`, and optional callbacks `on_agent_turn` / `on_participant_turn`.
