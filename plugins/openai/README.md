# OpenAI Plugin for Vision Agents

OpenAI LLM integration for Vision Agents framework with support for both standard and realtime interactions.

It enables features such as:
- Real-time transcription and language processing using OpenAI models
- Easy integration with other Vision Agents plugins and services
- Function calling capabilities for dynamic interactions

## Installation

```bash
pip install vision-agents[openai]
```

## Usage

### Standard LLM

This example shows how to use "gpt-4.1" model with TTS and STT services for audio communication via `openai.LLM()` API.

The `openai.LLM()` class uses OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) under the hood. 

To work with models via legacy [Chat Completions API](https://platform.openai.com/docs/api-reference/chat), see the [Chat Completions models](#chat-completions-models) section. 

```python
from vision_agents.core import User, Agent
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, cartesia, smart_turn, openai

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Friendly AI"),
    instructions="Be nice to the user",
    llm=openai.LLM("gpt-4.1"),
    tts=cartesia.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(),
)
```

### Realtime LLM

Realtime audio and video communication is also supported via `Realtime` class.
In this mode, the model handles audio and video processing directly without the need for TTS and STT services.

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, openai

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Friendly AI"),
    instructions="Be nice to the user",
    llm=openai.Realtime(),
)
```

### Chat Completions models
The `openai.ChatCompletionsLLM` and `openai.ChatCompletionsVLM` classes provide APIs for text and vision models that use the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat).  

They are compatible with popular inference backends such as vLLM, TGI, and Ollama. 

For example, you can use them to interact with Qwen 3 VL visual model hosted on [Baseten](https://www.baseten.co/):

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import deepgram, getstream, elevenlabs, vogent, openai

# Instantiate the visual model wrapper
llm = openai.ChatCompletionsVLM(model="qwen3vl")
    
# Create an agent with video understanding capabilities
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Video Assistant", id="agent"),
    instructions="You're a helpful video AI assistant. Analyze the video frames and respond to user questions about what you see.",
    llm=llm,
    stt=deepgram.STT(),
    tts=elevenlabs.TTS(),
    turn_detection=vogent.TurnDetection(),
    processors=[],
)
```

For full code, see [examples/qwen_vl_example](examples/qwen_vl_example/README.md).


## Function Calling

The `LLM` and `Realtime` APIs support function calling, allowing the assistant to invoke custom functions you define.   

This enables dynamic interactions like:

- Database queries
- API calls to external services
- File operations
- Custom business logic

```python
from vision_agents.plugins import openai

llm = openai.LLM("gpt-4.1")
# Or use openai.Realtime() for realtime model



@llm.register_function(
    name="get_weather",
    description="Get the current weather for a given city"
)
def get_weather(city: str) -> dict:
    """Get weather information for a city."""
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny"
    }
# The function will be automatically called when the model decides to use it
```

## Requirements
- Python 3.10+
- GetStream account for video calls
- Open AI API key

## Links
- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)

## License
MIT
