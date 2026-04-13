# Decart Plugin for Vision Agents

Decart integration for Vision Agents framework, enabling real-time video restyling capabilities.

It enables features such as:

- Real-time video transformation using generative AI models
- Dynamic style changing via prompts
- Seamless integration with Vision Agents video pipeline

## Installation

```bash
uv add "vision-agents[decart]"
# or directly
uv add vision-agents-plugins-decart
```

## Usage

This example shows how to use the `RestylingProcessor` to transform a user's video feed in real-time.

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini, decart

processor = decart.RestylingProcessor(
    initial_prompt="Studio Ghibli animation style",
    model="lucy_2_rt",
)

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Styled AI"),
    instructions="Be helpful",
    llm=gemini.Realtime(),
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

### Reference Images ("costumes")

For models like Lucy that accept a reference image, pass it at construction
time and/or swap it atomically with a prompt via `update_state`:

```python
processor = decart.RestylingProcessor(
    model="lucy_2_rt",
    initial_prompt="A person wearing a superhero costume",
    initial_image="./costumes/superhero.png",  # your own reference image
)

# Later — atomically change prompt + reference image
await processor.update_state(
    prompt="A person wearing a wizard robe",
    image="./costumes/wizard.png",
)

# Image-only update
await processor.update_state(image=b"<raw image bytes>")
```

`initial_image` and `update_state(image=...)` accept `bytes`, a local file
path, an `http(s)` URL, a `data:` URI, or a raw base64 string.

## Configuration

The plugin requires a Decart API key. You can provide it in two ways:

1. Set the environment variable `DECART_API_KEY`
2. Pass it directly to the constructor: `RestylingProcessor(api_key="...")`

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
