# OrcaRouter Plugin

This plugin provides LLM capabilities using OrcaRouter's API, an OpenAI-compatible model routing gateway that exposes models from several providers behind one endpoint and one key. It enables developers to switch between models from various providers (OpenAI, Anthropic, Google, DeepSeek, Qwen, MiniMax, and others) without changing their code.

## Features

- Access to multiple LLM providers through a single API
- OpenAI-compatible interface for easy integration
- Support for various models including GPT, Claude, Gemini, GLM, and more
- Named routers such as `orcarouter/auto` that pick an upstream per request
- Automatic conversion of instructions to system messages
- Manual conversation history management

## Installation

```bash
uv add "vision-agents[orcarouter]"
# or directly
uv add vision-agents-plugins-orcarouter
```

## Usage

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import orcarouter, getstream, elevenlabs, deepgram, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="OrcaRouter AI"),
    instructions="Be helpful and friendly to the user",
    llm=orcarouter.LLM(model="openai/gpt-4o-mini"),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(),
)
```

## Configuration

Set your API key:

```bash
export ORCAROUTER_API_KEY="sk-orca-..."
```

Or pass it directly:

```python
llm = orcarouter.LLM(api_key="sk-orca-...", model="anthropic/claude-opus-4.8")
```

## Models

Model ids carry a vendor namespace, for example:

- `openai/gpt-4o-mini` (default)
- `openai/gpt-5.5`
- `anthropic/claude-opus-4.8`
- `z-ai/glm-5.2`

`orcarouter/auto` is a named router rather than a model: it selects an upstream
per request. It is not the default because its candidate pool can include models
without tool calling. When it is used together with tools, this plugin sends a
fallback list of tool-capable models so tool calls keep working.

The full catalog is at https://www.orcarouter.ai/models.
