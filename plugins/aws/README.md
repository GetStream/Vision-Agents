# AWS Plugin for Vision Agents

AWS (Bedrock) LLM integration for Vision Agents framework with support for both standard and realtime interactions. Includes AWS Polly TTS.

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

### Realtime Text/Image Usage

Nova sonic audio realtime STS is also supported:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Story Teller AI"),
    instructions="Tell a story suitable for a 7 year old about a dragon and a princess",
    llm=aws.Realtime(),
)
```

### Polly TTS Usage

```python
from vision_agents.plugins import aws
from vision_agents.core.tts.manual_test import manual_tts_to_wav
import asyncio

async def main():
    # For PCM, AWS Polly supports 8000 or 16000 Hz
    tts = aws.TTS(voice_id="Joanna", sample_rate=16000)
    await manual_tts_to_wav(tts, sample_rate=16000, channels=1)

asyncio.run(main())
```

## Running the examples

Create a `.env` file, or cp .env.example to .env and fill in

```
STREAM_API_KEY=your_stream_api_key_here
STREAM_API_SECRET=your_stream_api_secret_here

AWS_BEARER_TOKEN_BEDROCK=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

FAL_KEY=
CARTESIA_API_KEY=
DEEPGRAM_API_KEY=
```
