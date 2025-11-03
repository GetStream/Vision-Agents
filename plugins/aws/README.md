# AWS Plugin for Vision Agents

AWS (Bedrock) LLM integration for Vision Agents framework with support for both standard and realtime interactions.

## Installation

```bash
pip install vision-agents-plugins-aws
```

## Usage

### Standard LLM Usage

This example shows how to use qwen3 on bedrock for the LLM.

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Friendly AI"),
    instructions="Be nice to the user",
    llm=aws.LLM(model="qwen.qwen3-32b-v1:0"),
    tts=cartesia.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(buffer_duration=2.0, confidence_threshold=0.5),
)
```

The full example is available in example/aws_qwen_example.py

### Realtime Audio Usage

Nova Sonic audio realtime STS is also supported:

```python    
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Story Teller AI"),
    instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
    llm=aws.Realtime(),
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

The Realtime implementation **fully supports** function calling with AWS Nova Sonic. Register functions using the `@llm.register_function` decorator:

```python
from vision_agents.plugins import aws

llm = aws.Realtime(
    model="amazon.nova-sonic-v1:0",
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

# The function will be automatically called when the model decides to use it
```

See `example/aws_realtime_function_calling_example.py` for a complete example.

## Running the examples

Create a `.env` file, or cp .env.example to .env and fill in

```
STREAM_API_KEY=your_stream_api_key_here
STREAM_API_SECRET=your_stream_api_secret_here

AWS_BEARER_TOKEN_BEDROCK=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

FAL_KEY=
CARTESIA_API_KEY=
DEEPGRAM_API_KEY=
```