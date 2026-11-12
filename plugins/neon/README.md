# Neon AI Gateway Plugin for Vision Agents

The Neon plugin adds [Neon AI Gateway](https://neon.com/docs/ai-gateway/overview) as an LLM provider for [Vision Agents](https://visionagents.ai). AI Gateway serves models from OpenAI, Anthropic, Meta, Alibaba and more behind a single Neon credential, with no provider API keys. It uses the gateway's OpenAI-compatible Chat Completions endpoint, so the implementation inherits the streaming and tool-calling behavior of the OpenAI Chat Completions plugin.

## Installation

```bash
uv add "vision-agents[neon]"
# or directly
uv add vision-agents-plugins-neon
```

## Configuration

Create a credential with the `ai_gateway:invoke` scope and copy your branch's AI Gateway host. Both are in the Neon Console under **Connect** > **AI Gateway**, or you can run:

```bash
neon credentials create --scope ai_gateway:invoke
```

```bash
export NEON_AI_GATEWAY_TOKEN=nt_live_...
export NEON_AI_GATEWAY_BASE_URL=https://br-xxx-api.ai.c-2.us-east-2.aws.neon.tech
```

`NEON_AI_GATEWAY_BASE_URL` is the bare branch host. The plugin appends `/v1`. See [Get started with Neon AI Gateway](https://neon.com/docs/ai-gateway/get-started) for details.

## Usage

```python
from vision_agents.plugins import neon

llm = neon.LLM()  # defaults to gpt-5-mini
llm = neon.LLM(model="claude-sonnet-4-6")
```

## Models

Use the short model ID in `model`. Some examples:

| Provider | Model ID |
|----------|----------|
| OpenAI | `gpt-5-mini` (default), `gpt-5-4`, `gpt-5-5` |
| Anthropic | `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7` |
| Alibaba | `qwen3-next-80b-a3b-instruct` |

See the [Neon AI Gateway model catalog](https://neon.com/ai-gateway#models) for the full list.

## Notes

- `gpt-5-3-codex` and `gpt-5-5-pro` only work with the Responses API and are not supported by this plugin.
- AI Gateway uses prepaid Neon credits and is available in select AWS regions. See [Model reference](https://neon.com/docs/ai-gateway/models).
