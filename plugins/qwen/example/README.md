# Qwen Realtime Example

This example demonstrates how to use Qwen Realtime with Vision Agents for real-time conversations.

## Features

- **Real-time streaming**: Direct audio streaming from Qwen Realtime API
- **No text input**: The model does not support text input, so start speaking once you join the call
- **Video support**: Configure frames per second for video processing

## Installation

```bash
uv add vision-agents[qwen]
```

## Quick Start

1. Set your API key in your environment:

```bash
export DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

Or create a `.env` file:

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

2. Run the example:

```bash
uv run python qwen_realtime_example.py
```

## Code Example

```python
from dotenv import load_dotenv
from vision_agents.core import Agent, User, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, qwen

load_dotenv()

async def create_agent(**kwargs) -> Agent:
    llm = qwen.Realtime(fps=1)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Qwen Assistant", id="agent"),
        instructions="You are a helpful AI assistant. Be friendly and conversational.",
        llm=llm,
    )
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.edge.open_demo(call)
        await agent.finish()

if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
```

## Configuration

### Environment Variables

- **`DASHSCOPE_API_KEY`**: Your DashScope/Alibaba API key (required)

### Realtime Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fps` | Video frames per second | `1` |
| `api_key` | DashScope API key | `None` (from env) |

## Requirements

- Python 3.10+
- DashScope API key
- Stream API credentials (configured via `getstream.Edge()`)
- `vision-agents` framework

## Notes

- The model is hosted in Singapore, so latency may vary depending on your location
- The model does not support text input - once you join the call, simply start speaking to the agent
- This example uses the CLI interface for easy interaction
