# Decart Plugin for Vision Agents

Decart integration for Vision Agents framework, enabling real-time video restyling capabilities.

It enables features such as:
- Real-time video transformation using generative AI models
- Dynamic style changing via prompts
- Seamless integration with Vision Agents video pipeline

## Installation

```bash
uv add vision-agents[decart]
```

## Usage

This example shows how to use the `RestylingProcessor` to transform a user's video feed in real-time.

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, openai, decart

# Initialize the restyling processor
processor = decart.RestylingProcessor(
    initial_prompt="A cute animated movie with vibrant colours",
    model="mirage_v2"
)

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Styled AI"),
    instructions="You are a helpful assistant.",
    llm=openai.LLM("gpt-4o-mini"),
    # Add the processor to the agent's pipeline
    processors=[processor],
)
```

### Dynamic Prompt Updates

You can register a function to update the style prompt dynamically based on the conversation:

```python
@llm.register_function(
    description="Change the video style prompt"
)
async def change_style(prompt: str) -> str:
    await processor.update_prompt(prompt)
    return f"Style changed to: {prompt}"
```

## Configuration

The plugin requires a Decart API key. You can provide it in two ways:

1. Set the environment variable `DECART_API_KEY`
2. Pass it directly to the constructor: `RestylingProcessor(api_key="...")`

## Links
- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
