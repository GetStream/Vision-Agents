"""LiteLLM plugin for Vision Agents.

Routes to 100+ LLM providers (OpenAI, Anthropic, Google, Azure, Bedrock,
Ollama, etc.) via the litellm SDK. No proxy server needed.

Model strings use the provider/model format, e.g.
anthropic/claude-sonnet-4-20250514, azure/gpt-4o, openai/gpt-4o.

See https://docs.litellm.ai/docs/providers for all supported models.
"""

from .litellm_llm import LiteLLMChatCompletions

__all__ = ["LiteLLMChatCompletions"]
