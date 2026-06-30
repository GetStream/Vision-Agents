# TwelveLabs Plugin

This plugin brings [TwelveLabs](https://twelvelabs.io) **Pegasus** video
understanding to Vision Agents as a first-class `VideoLLM`.

Unlike frame-by-frame VLMs, Pegasus analyzes a short **video clip**, so it can
reason about motion and events over time ("what just happened?") rather than a
single still frame. Recent frames from the watched track are buffered, encoded
into a short MP4 clip on demand, uploaded to the TwelveLabs Assets API, and
analyzed with your prompt. The streamed answer is vocalized by the agent's TTS
service.

## Installation

```bash
uv add "vision-agents[twelvelabs]"
# or directly
uv add vision-agents-plugins-twelvelabs
```

You can grab a free API key at https://twelvelabs.io — there's a generous free
tier.

## Quick Start

```python
import asyncio
import os
from dotenv import load_dotenv
from vision_agents.core import User, Agent, Runner
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, elevenlabs, twelvelabs
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = twelvelabs.PegasusVLM(
        api_key=os.getenv("TWELVELABS_API_KEY"),  # or set TWELVELABS_API_KEY
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(5)  # let a few seconds of video buffer
            await agent.simple_response("Describe what just happened in the video")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
```

## Configuration

### PegasusVLM Parameters

- `api_key`: str - TwelveLabs API key. If not provided, read from the
  `TWELVELABS_API_KEY` environment variable.
- `model_name`: str - Pegasus model identifier (default: `"pegasus1.5"`).
- `fps`: float - Frame sampling rate for the buffered clip (default: `1.0`).
- `clip_seconds`: int - Length of the clip analyzed per request. Pegasus
  requires at least 4 seconds of video (default: `5`).
- `max_tokens`: int - Maximum tokens in the response. Pegasus requires at least
  `512` (default: `512`).

## Notes

- Pegasus requires a minimum resolution of 360x360; lower-resolution frames are
  scaled up to that floor on encode.
- Pegasus requires the analyzed clip to be at least 4 seconds long, so
  `clip_seconds` must be `>= 4`.
- Each request uploads a clip and runs server-side analysis, so latency is
  higher than single-frame VLMs. Tune `fps` and `clip_seconds` for your use
  case.

## Testing

```bash
# Unit tests (no API key needed)
uv run pytest plugins/twelvelabs/tests -m "not integration"

# Integration tests (needs TWELVELABS_API_KEY)
export TWELVELABS_API_KEY="your-key-here"
uv run pytest plugins/twelvelabs/tests -m integration
```

## Links

- [TwelveLabs Documentation](https://docs.twelvelabs.io/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)
