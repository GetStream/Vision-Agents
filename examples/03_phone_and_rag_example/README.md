# Phone + RAG Example

A voice AI agent that answers phone calls via Twilio with RAG (Retrieval Augmented Generation) capabilities.

## RAG Backend Options

Configure via the `RAG_BACKEND` environment variable:

| Backend | Description |
|---------|-------------|
| `gemini` (default) | Uses Gemini's built-in File Search |
| `turbopuffer` | Uses TurboPuffer + LangChain with function calling |

### Gemini File Search

- Documents uploaded and indexed by Gemini
- Automatic retrieval during conversations
- No additional infrastructure needed

### TurboPuffer + LangChain

- More control over the RAG process
- Exposed as a callable function
- Works with any LLM that supports function calling

```python
@llm.register_function(description="Search knowledge base")
async def search_knowledge(query: str) -> str:
    return await rag.search(query, top_k=3)
```

## Setup

### Environment Variables

```bash
# Stream
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# Twilio
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token

# Ngrok (for exposing local server to Twilio)
NGROK_URL=your_ngrok_url  # e.g., abc123.ngrok.io

# AI Services
GOOGLE_API_KEY=your_google_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key

# RAG Backend (optional, defaults to "gemini")
RAG_BACKEND=gemini  # or "turbopuffer"

# For TurboPuffer backend only
OPENAI_API_KEY=your_openai_api_key
TURBO_PUFFER_KEY=your_turbopuffer_api_key
```

### Install & Run

```bash
cd examples/03_phone_and_rag_example
uv sync

# Run with Gemini (default)
uv run python phone_and_rag_example.py

# Run with TurboPuffer
RAG_BACKEND=turbopuffer uv run python phone_and_rag_example.py
```

### Configure Twilio

1. Get a Twilio phone number
2. Set webhook URL to `https://<your-ngrok-url>/twilio/voice` (POST)

## How It Works

1. **Incoming Call**: Twilio receives a call to your phone number
2. **Webhook**: Twilio sends a POST to `/twilio/voice`
3. **Media Stream**: Bidirectional WebSocket at `/twilio/media/{call_sid}`
4. **Audio Bridge**: Audio bridged between Twilio (mulaw @ 8kHz) and Stream
5. **AI Agent**: Agent uses RAG to answer questions about Stream's products

## Knowledge Base

The `knowledge/` directory contains product documentation:
- `chat.md` - Chat API
- `video.md` - Video API
- `feeds.md` - Feeds API
- `moderation.md` - Moderation features

## Using TurboPuffer RAG Independently

```python
from rag_turbopuffer import TurboPufferRAG, create_rag

# Quick setup
rag = await create_rag(namespace="my-knowledge", knowledge_dir="./knowledge")

# Search
results = await rag.search("How does the chat API work?")

# Register as LLM function
@llm.register_function(description="Search knowledge base")
async def search_knowledge(query: str) -> str:
    return await rag.search(query)
```

