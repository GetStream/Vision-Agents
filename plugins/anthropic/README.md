# Anthropic/Claude Plugin for Vision Agents

This plugin provides integration with Anthropic's Claude AI models for the Vision Agents framework.

## Installation

```bash
uv pip install vision-agents-plugins-anthropic
```

## Usage

```python
from vision_agents.plugins import anthropic

# Initialize the LLM
llm = anthropic.LLM(model="claude-3-5-sonnet-20241022")

# Simple response
response = await llm.simple_response("Explain quantum computing briefly")
print(response.text)

# Native API access
response = await llm.create_message(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1000
)
```

## Features

- **Streaming Support**: Real-time response streaming
- **Tool/Function Calling**: Multi-hop tool execution support
- **Conversation Memory**: Automatic conversation history tracking
- **Event System**: Integrated event broadcasting for monitoring

## Available Models

See [Anthropic's model documentation](https://docs.anthropic.com/en/docs/about-claude/models/overview) for the latest available models.

## Configuration

Set your API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or pass it directly:

```python
llm = anthropic.LLM(model="claude-3-5-sonnet-20241022", api_key="your-api-key")
```

## Documentation

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Vision Agents Documentation](https://visionagents.ai/)
