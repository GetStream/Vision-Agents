# OpenAI Realtime WebRTC Example

This example demonstrates how to use OpenAI's Realtime API for speech-to-speech conversation through WebRTC.

## Overview

The OpenAI Realtime API enables real-time, bidirectional audio conversations with GPT-4. This example shows how to:

- Create a WebRTC peer connection with OpenAI's servers
- Stream audio from participants to OpenAI for processing
- Stream OpenAI's audio responses back to participants
- Handle automatic turn detection for natural conversations

## Prerequisites

1. An OpenAI API key with access to the Realtime API
2. A Stream account with video/audio capabilities
3. Python 3.12 or higher

## Setup
1. Go to the example's directory
    ```bash
    cd examples/other_examples/openai_realtime_webrtc
    ```

2. Install dependencies:
    ```bash
    uv sync
    ```

3. Create a `.env` file with your credentials:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    STREAM_API_KEY=your_stream_api_key
    STREAM_API_SECRET=your_stream_api_secret
    ```

## Running the Example

```bash
uv run python openai_realtime_example.py
```

The script will:
1. Create a Stream video call
2. Open your browser with the Stream demo app
3. Start the OpenAI STS agent
4. Connect the agent to the call

Once connected, you can speak naturally with the AI assistant. The agent will:
- Listen to your speech
- Process it through OpenAI's Realtime model
- Respond with natural, synthesized speech

## Key Features

### Speech-to-Speech Mode
The agent operates in STS (Speech-to-Speech) mode, which means:
- No separate STT (Speech-to-Text) or TTS (Text-to-Speech) services are needed
- Audio is processed end-to-end by OpenAI's model
- Lower latency compared to traditional STT → LLM → TTS pipelines

### Automatic Turn Detection
The agent uses OpenAI's server-side voice activity detection (VAD) to:
- Detect when you start and stop speaking
- Automatically generate responses at appropriate times
- Handle interruptions naturally

### Voice Options
You can customize the assistant's voice by changing the `voice` parameter:
- `alloy` (default): Neutral and balanced
- `echo`: Warm and conversational
- `fable`: Expressive and animated
- `onyx`: Deep and authoritative
- `nova`: Friendly and upbeat
- `shimmer`: Clear and articulate

## Architecture

```
┌─────────────┐     Audio      ┌──────────────┐     WebRTC      ┌─────────────┐
│   Browser   │ ───────────────►│ Stream Call  │ ───────────────►│  OpenAI     │
│   (Human)   │                 │   (Agent)    │                 │  Realtime   │
│             │◄─────────────── │              │◄─────────────── │    API      │
└─────────────┘     Audio      └──────────────┘     Audio       └─────────────┘
```

The agent acts as a bridge between the Stream call and OpenAI's Realtime API:
1. Receives audio from call participants via Stream's WebRTC
2. Forwards audio to OpenAI via data channel events
3. Receives audio responses from OpenAI
4. Plays responses back to call participants

## Customization

You can customize the agent by modifying:

- **Instructions**: Change the `instructions` parameter to define the assistant's personality and behavior
- **Model**: Update the `model` parameter (when new models are released)
- **Turn Detection**: Disable automatic turn detection for push-to-talk scenarios
- **System Prompt**: Provide detailed system instructions for complex use cases

## Troubleshooting

1. **No audio from agent**: Ensure your OpenAI API key has access to the Realtime API
2. **Connection errors**: Check firewall settings - WebRTC requires UDP connectivity
3. **High latency**: Ensure you're geographically close to OpenAI's servers
4. **Audio quality issues**: Use a good quality microphone and stable internet connection

## Learn More

- [OpenAI Realtime API Documentation](https://platform.openai.com/docs/guides/realtime)
- [Stream Video & Audio Documentation](https://getstream.io/video/docs/)
- [WebRTC Fundamentals](https://webrtc.org/)
