# Simple Agent Example

This example shows you how to build a basic video AI agent using [Vision Agents](https://visionagents.ai/). The agent
can have conversations with users through video and voice input and output.

- Listens to user speech and converts it to text
- Processes the conversation using an LLM (Large Language Model)
- Responds with natural-sounding speech
- Runs on Stream's low-latency edge network

## Prerequisites

- Python 3.13 or higher
- API keys for:
    - [OpenAI](https://openai.com) (for the LLM)
    - [Elevenlabs](https://elevenlabs.io/) (for text-to-speech)
    - [Deepgram](https://deepgram.com/) (for speech-to-text)
    - [Stream](https://getstream.io/) (for video/audio infrastructure)
    - [Smart Turn](https://fal.ai/models/fal-ai/smart-turn) (for turn detection)

## Installation

1. Go to the example's directory
    ```bash
    cd examples/01_simple_agent_example
    ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ELEVENLABS_API_KEY=your_11labs_key
   DEEPGRAM_API_KEY=your_deepgram_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   ```

## Running the Example

Run the agent:

```bash
uv run simple_agent_example.py run
```

The agent will:

1. Create a video call
2. Open a demo UI in your browser
3. Join the call and start listening
4. Respond to your voice input

## Code Walkthrough

### Setting Up the Agent

The code creates an agent with several components:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="My happy AI friend", id="agent"),
    instructions="You're a video AI assistant...",
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(),
)
```

**Components:**

- `edge`: Handles low-latency audio/video transport
- `agent_user`: Sets the agent's name and ID
- `instructions`: Tells the agent how to behave
- `llm`: The language model that powers the conversation
- `tts`: Converts agent responses to speech
- `stt`: Converts user speech to text
- `turn_detection`: Detects when the user has finished speaking

### Creating and Joining a Call

```python
call = agent.edge.client.video.call("default", str(uuid4()))

async with agent.join(call):
    await agent.finish()
```

This code:

1. Creates a new video call with a unique ID
2. Has the agent join the call
3. Keeps the agent running until the call ends

**Note:** The CLI automatically opens the demo UI by default. Use `--no-demo` flag to disable it.

### Alternative: Using Realtime LLMs

You can simplify the setup by using a realtime LLM like OpenAI Realtime or Gemini Live. These models handle
speech-to-text and text-to-speech internally:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="My happy AI friend", id="agent"),
    instructions="You're a video AI assistant...",
    llm=openai.Realtime()
)
# No need for separate tts, stt, or vad components
```

### Native API Access

Vision Agents gives you direct access to native LLM APIs. You can use OpenAI's `create_response` method or any other
provider-specific features:

```python
await llm.create_response(input=[
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Tell me a poem"},
            {"type": "input_image", "image_url": "https://..."}
        ]
    }
])
```

## Customization

### Change the Instructions

Edit the `instructions` parameter to change how your agent behaves. The instructions file (`instructions.md`) shows an
example of adding personality.

### Use Different Models

You can swap out any component:

- Try `openai.Realtime()` for lower latency
- Use `gemini.Realtime()` for Google's model
- Switch TTS providers to `elevenlabs.TTS()` or `kokoro.TTS()`

### Add Processors

Add items to the `processors` list to give your agent new capabilities. See the golf coach example for how to use YOLO
for object detection.

## Next Steps

- Check out the [golf coach example](../02_golf_coach_example) to learn about video processing
- Read the [Vision Agents documentation](https://visionagents.ai) for more features
- Explore other examples in the `examples` directory

## Learn More

- [Building a Voice AI app](https://visionagents.ai/introduction/voice-agents)
- [Building a Video AI app](https://visionagents.ai/introduction/video-agents)
- [Main Vision Agents README](../../README.md)

