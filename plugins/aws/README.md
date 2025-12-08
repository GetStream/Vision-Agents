# AWS Plugin for Vision Agents

AWS (Bedrock) integration for Vision Agents framework with support for standard LLM, realtime with Nova Sonic, and text-to-speech with automatic session resumption.

## Installation

```bash
uv add vision-agents[aws]
```

## Usage

### Standard LLM Usage

The AWS plugin supports various Bedrock models including Qwen, Claude, and others. Claude models also support vision/image inputs.

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import aws, getstream, cartesia, deepgram, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Friendly AI"),
    instructions="Be nice to the user",
    llm=aws.LLM(
        model="qwen.qwen3-32b-v1:0",
        region_name="us-east-1"
    ),
    tts=cartesia.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(buffer_duration=2.0, confidence_threshold=0.5),
)
```

For vision-capable models like Claude:

```python
llm = aws.LLM(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Send image with text
response = await llm.converse(
    messages=[{
        "role": "user",
        "content": [
            {"image": {"format": "png", "source": {"bytes": image_bytes}}},
            {"text": "What do you see in this image?"}
        ]
    }]
)
```

### Realtime Audio Usage

AWS Nova 2 Sonic provides realtime speech-to-speech capabilities with automatic reconnection logic. The default model is `amazon.nova-2-sonic-v1:0`.

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import aws, getstream

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Story Teller AI"),
    instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
    llm=aws.Realtime(
        model="amazon.nova-2-sonic-v1:0",
        region_name="us-east-1",
        voice_id="matthew"  # See available voices in AWS Nova documentation
    ),
)
```

The Realtime implementation includes automatic reconnection logic that reconnects after periods of silence or when approaching connection time limits.

See `example/aws_realtime_nova_example.py` for a complete example.

### Text-to-Speech (TTS)

AWS Polly TTS is available for converting text to speech:

```python
from vision_agents.plugins import aws

tts = aws.TTS(
    region_name="us-east-1",
    voice_id="Joanna",  # AWS Polly voice ID
    engine="neural",  # 'standard' or 'neural'
    text_type="text",  # 'text' or 'ssml'
    language_code="en-US"
)

# Use in agent
agent = Agent(
    llm=aws.LLM(model="qwen.qwen3-32b-v1:0"),
    tts=tts,
    # ... other components
)
```

## Function Calling

### Standard LLM (aws.LLM)

The standard LLM implementation **fully supports** function calling. Register functions using the `@llm.register_function` decorator:

```python
from vision_agents.plugins import aws

llm = aws.LLM(
    model="qwen.qwen3-32b-v1:0",
    region_name="us-east-1"
)

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
```

### Realtime (aws.Realtime)

The Realtime implementation **fully supports** function calling with AWS Nova 2 Sonic. Register functions using the `@llm.register_function` decorator:

```python
from vision_agents.plugins import aws

llm = aws.Realtime(
    model="amazon.nova-2-sonic-v1:0",
    region_name="us-east-1",
    voice_id="matthew"
)

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

See `example/aws_realtime_function_calling_example.py` for a complete example.

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```
STREAM_API_KEY=your_stream_api_key_here
STREAM_API_SECRET=your_stream_api_secret_here

AWS_BEDROCK_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

CARTESIA_API_KEY=
DEEPGRAM_API_KEY=
```

Make sure your `.env` file is configured before running the examples.
