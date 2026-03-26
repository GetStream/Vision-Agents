## Gemini Live Speech-to-Speech Plugin

Google Gemini Live Speech-to-Speech (STS) plugin for GetStream. It connects a realtime Gemini Live session to a Stream video call so your assistant can speak and listen in the same call.

## Installation

```bash
uv add "vision-agents[gemini]"
# or directly
uv add vision-agents-plugins-gemini
```

### Requirements

- **Python**: 3.10+
- **Dependencies**: `getstream[webrtc"]`, `getstream-plugins-common`, `google-genai>=1.51.0`
- **API key**: `GOOGLE_API_KEY` or `GEMINI_API_KEY` set in your environment

### Quick Start

Below is a minimal example that attaches the Gemini Live output audio track to a Stream call and streams microphone audio into Gemini. The assistant will speak back into the call, and you can also send text messages to the assistant.

```python
from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI coach"),
        instructions="Read @coaching.md",
        llm=gemini.Realtime(model="gemini-3.1-flash-live-preview"),
        processors=[],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.llm.simple_response(
            text="Say hi. After the user joins ask them about their day"
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
```

Video frames from remote participants are forwarded to Gemini automatically when `fps` is set and the model supports it:

```python
llm=gemini.Realtime(fps=3)  # forward video at 3 frames per second
```

The `Agent` subscribes to track events internally, so no manual wiring is needed.
For a full runnable example, see `examples/02_golf_coach_example/golf_coach_example.py`.

### Gemini Vision (VLM)

Use Gemini 3 vision models with the Agent API (video frames are forwarded
automatically when the call has active video).

```python
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream


async def create_agent(**kwargs) -> Agent:
    vlm = gemini.VLM(model="gemini-3-flash-preview")
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Gemini Vision Agent", id="gemini-vision-agent"),
        instructions="Describe what you see in one sentence.",
        llm=vlm,
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.finish()


Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
```

Key configuration knobs for `GeminiVLM`: `fps`, `frame_buffer_seconds`,
`thinking_level`, `media_resolution`. For a full example, see
`plugins/gemini/example/gemini_vlm_agent_example.py`.

### Features

- **Bidirectional audio**: Streams microphone PCM to Gemini, and plays Gemini speech into the call using `output_track`.
- **Video frame forwarding**: Sends remote participant video frames to Gemini Live for multimodal understanding. Use `start_video_sender` with a remote `MediaStreamTrack`.
- **Text messages**: Use `send_text` to add text turns directly to the conversation.
- **Barge-in (interruptions)
  **: When the user starts speaking, current playback is interrupted so Gemini can focus on the new input. Playback automatically resumes after brief silence.
- **Auto resampling**: `send_audio_pcm` will resample input frames to the target rate when needed.
- **Events**: Subscribe to `"audio"` for synthesized audio chunks and `"text"` for assistant text.

### API Overview

- **`GeminiLive(api_key: str | None = None, model: str = "gemini-live-2.5-flash-preview", config: LiveConnectConfigDict | None = None)`**: Create a new Gemini Live session. If
  `api_key` is not provided, the plugin reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` from the environment.
- **`GeminiVLM(model: str = "gemini-3-flash-preview", fps: int = 1, frame_buffer_seconds: int = 10, ...)`
  **: Vision-language model that buffers video frames and sends them with prompts.
- **`output_track`**: An `AudioStreamTrack` you can publish in your call via `add_tracks(audio=...)`.
- **`await send_text(text: str)`**: Send a user text message to the current turn.
- **`await send_audio_pcm(pcm: PcmData, target_rate: int = 48000)`**: Stream PCM frames to Gemini. Frames are converted to the required format and resampled if necessary.
- **`await wait_until_ready(timeout: float | None = None) -> bool`**: Wait until the underlying live session is connected.
- **`await interrupt_playback()` / `resume_playback()`**: Manually stop or resume synthesized audio playback. Useful if you want to manage barge-in behavior yourself.
- **`await start_video_sender(track: MediaStreamTrack, fps: int = 1)`**: Start forwarding video frames from a remote `MediaStreamTrack` to Gemini Live at the given frame rate.
- **`await stop_video_sender()`**: Stop the background video sender task, if running.
- **`await close()`**: Close the session and background tasks.

### Environment Variables

- **`GOOGLE_API_KEY` / `GEMINI_API_KEY`**: Gemini API key. One must be set.
- **`GEMINI_LIVE_MODEL`**: Optional override for the model name if you need a different variant.


### Troubleshooting

- **No audio playback**: Ensure you publish `output_track` to your call and the call is subscribed to the assistant's audio.
- **No responses**: Verify `GOOGLE_API_KEY`/`GEMINI_API_KEY` is set and has access to the chosen model. Try a different model via `model=`.
- **Sample-rate issues**: Use `send_audio_pcm(..., target_rate=48000)` to normalize input frames.

### Migration from Gemini 2.5

When migrating to Gemini 3:

- **Thinking**: If you were using complex prompt engineering (like Chain-of-thought) with Gemini 2.5, try Gemini 3 with `thinking_level="high"` and simplified prompts.
- **Temperature**: If your code explicitly sets temperature to low values, consider removing it and using the Gemini 3 default (1.0) to avoid potential looping issues.
- **PDF & Document Understanding**: Default OCR resolution for PDFs has changed. Test with `media_resolution="high"` if you need dense document parsing.
- **Token Consumption**: Gemini 3 defaults may increase token usage for PDFs but decrease for video. If requests exceed context limits, explicitly reduce `media_resolution`.
