# Qwen3-VL hosted on Baseten 
Qwen3-VL is the latest open-source Video Language Model (VLM) from Alibaba. 
This plugin allows developers to easily run the model hosted on [Baseten](https://www.baseten.co/) with Vision Agents. 
The model accepts text and video and responds with text vocalised with the TTS service of your choice. 

## Features

- **Video understanding**: Automatically buffers and forwards video frames to Baseten-hosted VLM models
- **Streaming responses**: Supports streaming text responses with real-time chunk events
- **Frame buffering**: Configurable frame rate and buffer duration for optimal performance
- **Event-driven**: Emits LLM events (chunks, completion, errors) for integration with other components

## Installation

```bash
uv add vision-agents[openai]
```

## Quick Start

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import openai, getstream, deepgram, elevenlabs, vogent

async def create_agent(**kwargs) -> Agent:
    # Initialize the Baseten VLM
    # The api key and base url can be passed via OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
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
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)
    
    async with agent.join(call):
        # The agent will automatically process video frames and respond to user input
        await agent.finish()
```

## Configuration

### Environment Variables

- **`OPENAI_API_KEY`**: Your Baseten API key (required)
- **`OPENAI_BASE_URL`**: The base URL for your Baseten API endpoint (required)

### Initialization Parameters

```python
openai.ChatCompletionsVLM(
    model: str,                    # Baseten model name (e.g., "qwen3vl")
    api_key: Optional[str] = None,  # API key (defaults to OPENAI_API_KEY env var)
    base_url: Optional[str] = None, # Base URL (defaults to OPENAI_BASE_URL env var)
    fps: int = 1,                   # Frames per second to process (default: 1)
    frame_buffer_seconds: int = 10, # Seconds of video to buffer (default: 10)
    client: Optional[AsyncOpenAI] = None,  # Custom OpenAI client (optional)
)
```

### Parameters

- **`model`**: The name of the Baseten-hosted model to use. Must be a vision-capable model.
- **`api_key`**: Your Baseten API key. If not provided, reads from `OPENAI_API_KEY` environment variable.
- **`base_url`**: The base URL for Baseten API. If not provided, reads from `OPENAI_BASE_URL` environment variable.
- **`fps`**: Number of video frames per second to capture and send to the model. Lower values reduce API costs but may miss fast-moving content. Default is 1 fps.
- **`frame_buffer_seconds`**: How many seconds of video to buffer. Total buffer size = `fps * frame_buffer_seconds`. Default is 10 seconds.
- **`client`**: Optional pre-configured `AsyncOpenAI` client. If provided, `api_key` and `base_url` are ignored.

## How It Works

1. **Video Frame Buffering**: The plugin automatically subscribes to video tracks when the agent joins a call. It buffers frames at the specified FPS for the configured duration.

2. **Frame Processing**: When responding to user input, the plugin:
   - Converts buffered video frames to JPEG format
   - Resizes frames to 800x600 (maintaining aspect ratio)
   - Encodes frames as base64 data URLs

3. **API Request**: Sends the conversation history (including system instructions) along with all buffered frames to the Baseten model.

4. **Streaming Response**: Processes the streaming response and emits events for each chunk and completion.

## Events

The plugin emits the following events:

- **`LLMResponseChunkEvent`**: Emitted for each text chunk in the streaming response
- **`LLMResponseCompletedEvent`**: Emitted when the response stream completes
- **`LLMErrorEvent`**: Emitted if an API request fails

## Requirements

- Python 3.10+
- `openai>=2.5.0`
- `vision-agents` (core framework)
- Baseten API key and base URL

## Notes

- **Frame Rate**: The default FPS of 1 is optimized for VLM use cases. Higher FPS values will increase API costs and latency.
- **Frame Size**: Frames are automatically resized to 800x600 pixels while maintaining aspect ratio to optimize API payload size.
- **Buffer Duration**: The 10-second default buffer provides context for the model while keeping memory usage reasonable.
- **Tool Calling**: Tool/function calling support is not yet implemented (see TODOs in code).

## Troubleshooting

- **No video processing**: Ensure the agent has joined a call with video tracks available. The plugin automatically subscribes to video when tracks are added.
- **API errors**: Verify your `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set correctly and the model name is valid.
- **High latency**: Consider reducing `fps` or `frame_buffer_seconds` to decrease the number of frames sent per request.
