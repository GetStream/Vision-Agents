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
Turn detection is integrated via a simple, unified `TurnDetection` protocol that the `Agent` understands. Implementations provide event-based turn management:

```python
from turn_detection import BaseTurnDetector, TurnEvent
from getstream.video.rtc.track_util import PcmData

class MyTurnDetector(BaseTurnDetector):
    async def process_audio(self, audio_data: PcmData, user_id: str, metadata: dict = None):
        # Analyze audio and emit events when turns change
        # self._emit_turn_event(TurnEvent.TURN_STARTED, event_data)
        pass

turn_detection = MyTurnDetector(mini_pause_duration=0.5, max_pause_duration=2.0)


agent = Agent(
    stt=your_stt_service,
    llm=llm,
    tts=your_tts_service,
    turn_detection=turn_detection,
)
```

The `TurnDetection` protocol requires: `start()`, `stop()`, `is_detecting()`, `process_audio()`, and event emission (`on`/`emit` from EventEmitter). The Agent automatically calls `process_audio()` with `PcmData` from Stream and listens for turn detection events.
