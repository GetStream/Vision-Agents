# NVIDIA VLM Example

This example demonstrates how to use the NVIDIA VLM plugin with Vision Agents to create a video-aware AI assistant.

## Features

- Video understanding: Automatically buffers and forwards video frames to NVIDIA VLM
- Real-time responses: Processes user speech and responds with video context
- Streaming: Supports streaming text responses

## Prerequisites

1. NVIDIA API key - Get one from [NVIDIA API](https://build.nvidia.com/)
2. Stream API credentials - Get from [GetStream](https://getstream.io/)
3. Deepgram API key - Get from [Deepgram](https://deepgram.com/)
4. ElevenLabs API key - Get from [ElevenLabs](https://elevenlabs.io/)

## Setup

1. Install dependencies:

```bash
cd plugins/nvidia/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
NVIDIA_API_KEY=your_nvidia_api_key
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Running the Example

```bash
uv run main.py run
```

The agent will:

1. Join a video call
2. Automatically buffer video frames
3. Respond to user questions about what it sees in the video
4. Process speech-to-text and text-to-speech in real-time

## Usage

Once the agent joins the call:

- Speak to the agent and ask questions about what it sees
- The agent will analyze the video frames and respond
- Example questions: "What do you see?", "Describe the scene", "What's happening?"

## Configuration

You can customize the VLM settings in `main.py`:

```python
llm = nvidia.VLM(
    model="nvidia/cosmos-reason2-8b",
    fps=1,  # Frames per second to buffer
    frame_buffer_seconds=10,  # Seconds of video to buffer
    frame_width=800,  # Frame width
    frame_height=600,  # Frame height
)
```
