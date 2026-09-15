## Gemini Plugin

Google Gemini integrations for Vision Agents, including Live speech-to-speech, streaming speech-to-text, LLM, and VLM support.

## Installation

```bash
uv add "vision-agents[gemini]"
# or directly
uv add vision-agents-plugins-gemini
```

### Requirements

- **Python**: 3.10+
- **Dependencies**: `vision-agents`, `google-genai>=2.19.0`
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
        llm=gemini.Realtime(),
        processors=[],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response(
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

The default Live model is `gemini-3.8-live` (latency-optimized audio). For background reasoning and async tools, use Extended Thinking:

```python
from vision_agents.plugins.gemini import LIVE_EXTENDED_THINKING_MODEL, Realtime

llm = Realtime(model=LIVE_EXTENDED_THINKING_MODEL)
```

`turn_complete` only ends a streaming chunk. Agent turn-complete events wait for `interaction_status=IDLE` (or the deprecated `REQUIRES_ACTION` alias) so thinking and async tool calls can continue. Tools are declared `NON_BLOCKING` by default; `blocking=True` is allowed only on `gemini-3.8-live`. Use `send_client_content(..., turn_complete=True)` to inject structured turns; that interrupts ongoing generation.

The `Agent` subscribes to track events internally, so no manual wiring is needed.
For full runnable examples, see `plugins/gemini/example/gemini_realtime_example.py`, `plugins/gemini/example/gemini_live_standalone_example.py` (no Stream SFU), and `examples/02_golf_coach_example/golf_coach_example.py`.

### Gemini Speech-to-Text

Use Gemini's Live transcription model in a standard STT, LLM, and TTS agent:

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import elevenlabs, gemini, getstream


async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="gemini-stt-agent"),
        instructions="Be helpful and concise.",
        stt=gemini.STT(),
        llm=gemini.LLM(),
        tts=elevenlabs.TTS(),
    )
```

Gemini automatically detects the spoken language by default. You can provide
BCP-47 language codes and vocabulary that should be recognized accurately:

```python
stt = gemini.STT(
    language_codes=["en-US"],
    custom_vocabulary=["Vision Agents", "GetStream"],
)
```

`gemini.STT` defaults to the Live model `gemini-3.5-transcribe-live`. It
streams 16 kHz PCM audio to Gemini and emits standard partial, final, and turn
events for the Vision Agents pipeline. Google also publishes
`gemini-3.5-transcribe` for unary/file transcription; this plugin uses the
Live model. For a full runnable example, see
`plugins/gemini/example/gemini_stt_example.py`.

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

- **Bidirectional audio**: The Agent streams call audio into Gemini Live and publishes Gemini speech back into the call.
- **Video**: Set `fps=` on `gemini.Realtime()` to forward remote participant frames. Default turn coverage is `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO`.
- **Text**: Use `agent.simple_response(text=...)` or `await llm.send_client_content(..., turn_complete=True)` (the latter interrupts ongoing generation).
- **Tools**: Function declarations default to `NON_BLOCKING`. `blocking=True` is allowed only on `gemini-3.8-live`.
- **Barge-in**: User speech interrupts the current agent turn via the Agent interrupt path (`await llm.interrupt()`).

### API Overview

- **`gemini.Realtime(model: str = "gemini-3.8-live", blocking: bool = False, thinking_level: ThinkingLevel | None = None, config: LiveConnectConfigDict | None = None, fps: int = 1, ...)`**: Live speech-to-speech. Reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` when `api_key` is omitted. Use `LIVE_EXTENDED_THINKING_MODEL` (`gemini-3.8-live-extended-thinking`) for background reasoning; that model applies `thinking_level=HIGH` when `thinking_config` is omitted and rejects `blocking=True`.
- **`gemini.VLM(model: str = "gemini-3-flash-preview", fps: int = 1, frame_buffer_seconds: int = 10, ...)`**: Vision-language model that buffers video frames and sends them with prompts.
- **`await simple_response(text)`**: Send a text instruction over the Live session.
- **`await send_client_content(turns, turn_complete=True)`**: Inject structured turns. `turn_complete=True` interrupts generation.
- **`await interrupt()`**: Stop the current agent turn.
- **`await watch_video_track(track)` / `await stop_watching_video_track()`**: Low-level video forwarding; the Agent calls these when `fps` is set.
- **`await close()`**: Close the session and background tasks.

### Environment Variables

- **`GOOGLE_API_KEY` / `GEMINI_API_KEY`**: Gemini API key. One must be set.
- **`GEMINI_LIVE_MODEL`**: Optional model override used by `plugins/gemini/example/gemini_realtime_example.py`.

### Troubleshooting

- **No audio playback**: Confirm the Agent joined the call (`async with agent.join(call)`) so tracks are published automatically.
- **Model unavailable**: The project behind the API key must have access to the Live model. Pass `model=` for an older Live model such as `gemini-3.1-flash-live-preview`, or use `LIVE_EXTENDED_THINKING_MODEL`.
- **No responses**: Verify `GOOGLE_API_KEY` / `GEMINI_API_KEY` is set. Extended Thinking requires `thinking_level` (the plugin sets `HIGH` by default).

### Migration notes

Gemini Live 3.8:

- Default model is `gemini-3.8-live` (was `gemini-3.1-flash-live-preview`). Pass `model=` to stay on an older Live model.
- Agent turn completion follows `interaction_status=IDLE`, not `turn_complete` alone.
- Requires `google-genai>=2.19.0`.

LLM / VLM (Gemini 3):

- **Thinking**: Prefer `thinking_level="high"` and simplified prompts over chain-of-thought prompt engineering.
- **Temperature**: If you set a low temperature, try the Gemini 3 default (1.0) to avoid looping.
- **PDF & documents**: Default OCR resolution changed. Use `media_resolution="high"` for dense documents.
- **Token usage**: Defaults may increase PDF tokens and decrease video tokens. Reduce `media_resolution` if you hit context limits.
