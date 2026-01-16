# xAI Realtime Voice Agent Example

This example demonstrates how to build a real-time voice conversation AI using xAI's Grok Voice Agent API with Vision Agents.

## Features

- Real-time voice conversations with xAI's Grok model
- Server-side voice activity detection (VAD)
- Multiple voice options (Ara, Rex, Sal, Eve, Leo)
- **Web search** - enabled by default, search the web for current information
- **X search** - enabled by default, search X (Twitter) for posts and information
- Function calling support

## Prerequisites

- Python 3.10 or higher
- API keys for:
  - [xAI](https://x.ai/) - for Grok voice API
  - [Stream](https://getstream.io/) - for audio infrastructure

## Installation

1. From the workspace root, ensure dependencies are installed:
    ```bash
    cd /path/to/Vision-Agents
    uv sync
    ```

2. Create a `.env` file in the workspace root with your API keys:
   ```
   XAI_API_KEY=your_xai_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   ```

## Running the Example

From the workspace root:

```bash
uv run plugins/xai/example/xai_realtime_example.py
```

## Voice Options

xAI provides 5 different voice options:

| Voice | Type | Tone |
|-------|------|------|
| **Ara** | Female | Warm, friendly (default) |
| **Rex** | Male | Confident, clear |
| **Sal** | Neutral | Smooth, balanced |
| **Eve** | Female | Energetic, upbeat |
| **Leo** | Male | Authoritative |

To change the voice:

```python
llm=xai.Realtime(
    voice="Rex",
    sample_rate=24000,
)
```

## Search Tools

By default, web search and X search are enabled. You can configure or disable them:

```python
# Disable both search tools
llm=xai.Realtime(
    web_search=False,
    x_search=False,
)

# Restrict X search to specific handles
llm=xai.Realtime(
    x_search_allowed_handles=["elonmusk", "xai"],
)
```

## Learn More

- [xAI Voice Agent API Documentation](https://docs.x.ai/docs/guides/voice/agent)
- [Vision Agents Documentation](https://visionagents.ai/)
