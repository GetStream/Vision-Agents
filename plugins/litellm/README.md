# vision-agents-plugins-litellm

LiteLLM plugin for [Vision Agents](https://github.com/GetStream/Vision-Agents), enabling access to 100+ LLM providers through a single unified interface.

## Installation

```bash
pip install vision-agents-plugins-litellm
```

## Usage

```python
from vision_agents.plugins.litellm import LiteLLMChatCompletions

# Use any litellm model string
llm = LiteLLMChatCompletions(model="anthropic/claude-sonnet-4-20250514")
llm = LiteLLMChatCompletions(model="azure/gpt-4o", api_key="...")
llm = LiteLLMChatCompletions(model="bedrock/anthropic.claude-3-haiku")
```

LiteLLM reads provider API keys from environment variables automatically (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).

See https://docs.litellm.ai/docs/providers for all supported models.
