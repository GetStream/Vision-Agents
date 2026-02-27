# Mistral Voxtral STT Plugin

Mistral Voxtral realtime speech-to-text integration for Vision Agents.

## Features

- Real-time speech recognition via WebSocket streaming
- Low-latency transcription using Voxtral models
- Automatic language detection
- Partial transcript streaming for responsive UX
- Sentence-level final transcripts (triggered by `.`, `?`, `!`)

## Installation

```bash
uv add vision-agents[mistral]
```

## Usage

```python
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, mistral


async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        stt=mistral.STT(),
        tts=deepgram.TTS(),
        llm=gemini.LLM(),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.run()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
```

Run with:

```bash
uv run plugins/mistral/example/mistral_stt_example.py run
```

## Turn Detection

Mistral Voxtral STT does not include built-in turn detection (`turn_detection=False`). You'll need to pair it with an external turn detection plugin.

## Configuration

| Parameter     | Description                                              | Default                                 |
|---------------|----------------------------------------------------------|-----------------------------------------|
| `api_key`     | Mistral API key                                          | `MISTRAL_API_KEY` env var               |
| `model`       | Model identifier                                         | `voxtral-mini-transcribe-realtime-2602` |
| `sample_rate` | Audio sample rate (Hz): 8000, 16000, 22050, 44100, 48000 | `16000`                                 |
| `client`      | Pre-configured Mistral client                            | `None`                                  |

## Events

The plugin emits standard STT events:

- `STTTranscriptEvent`: Final transcript (emitted at sentence boundaries or stream end)
- `STTPartialTranscriptEvent`: Partial word/delta as transcription streams

## Dependencies

- `mistralai[realtime]>=1.12.0`
- `vision-agents`
