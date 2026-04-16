# XAI Plugin for Stream Agents

This package provides xAI (Grok) integration for the Stream Agents ecosystem, enabling you to use xAI's powerful language models in your conversational AI applications.

## Features

- **Native xAI SDK Integration**: Full access to xAI's chat completion and streaming APIs
- **Conversation Memory**: Automatic conversation history management
- **Streaming Support**: Real-time response streaming with standardized events
- **Multimodal Support**: Handle text and image inputs
- **Event System**: Subscribe to response events for custom handling
- **Easy Integration**: Drop-in replacement for other LLM providers

## Installation

```bash
uv add "vision-agents[xai]"
# or directly
uv add vision-agents-plugins-xai
```

## Quick Start

```python
import asyncio
from vision_agents.plugins import xai

async def main():
    # Initialize with your xAI API key
    llm = xai.LLM(
        model="grok-4",
        api_key="your_xai_api_key"  # or set XAI_API_KEY environment variable
    )

    # Simple response
    response = await llm.simple_response("Explain quantum computing in simple terms")

    print(f"\n\nComplete response: {response.text}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Conversation with Memory

```python
from vision_agents.plugins import xai

llm = xai.LLM(model="grok-4", api_key="your_api_key")

# First message
await llm.simple_response("My name is Alice and I have 2 cats")

# Second message - the LLM remembers the context
response = await llm.simple_response("How many pets do I have?")
print(response.text)  # Will mention the 2 cats
```

### Using Instructions

```python
llm = LLM(
    model="grok-4",
    api_key="your_api_key"
)

# Create a response with system instructions
response = await llm.create_response(
    input="Tell me about the weather",
    instructions="You are a helpful weather assistant. Always be cheerful and optimistic.",
    stream=True
)
```

### Multimodal Input

```python
# Handle complex multimodal messages
advanced_message = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What do you see in this image?"},
            {"type": "input_image", "image_url": "https://example.com/image.jpg"},
        ],
    }
]

messages = LLM._normalize_message(advanced_message)
# Use with your conversation system
```

## API Reference

### XAILLM Class

#### Constructor

```python
LLM(
    model: str = "grok-4",
    api_key: Optional[str] = None,
    client: Optional[AsyncClient] = None
)
```

**Parameters:**

- `model`: xAI model to use (default: "grok-4")
- `api_key`: Your xAI API key (default: reads from `XAI_API_KEY` environment variable)
- `client`: Optional pre-configured xAI AsyncClient

#### Methods

##### `async simple_response(text: str, processors=None, participant=None)`

Generate a simple response to text input.

**Parameters:**

- `text`: Input text to respond to
- `processors`: Optional list of processors for video/voice AI context
- `participant`: Optional participant object

**Returns:** `LLMResponseEvent[Response]` with the generated text

##### `async create_response(input: str, instructions: str = "", model: str = None, stream: bool = True)`

Create a response with full control over parameters.

**Parameters:**

- `input`: Input text
- `instructions`: System instructions for the model
- `model`: Override the default model
- `stream`: Whether to stream the response (default: True)

**Returns:** `LLMResponseEvent[Response]` with the generated text

## Configuration

### Environment Variables

- `XAI_API_KEY`: Your xAI API key (required if not provided in constructor)

## Text-to-Speech (TTS)

The plugin also ships an `xai.TTS` class powered by [xAI's Grok Voice API](https://docs.x.ai/docs/guides/voice/tts). It provides five expressive voices with inline speech tags for fine-grained delivery control.

### Usage

```python
from vision_agents.plugins import xai

# Default voice (eve) — energetic, upbeat
tts = xai.TTS()

# Specify a voice
tts = xai.TTS(voice="ara")   # warm, friendly
tts = xai.TTS(voice="leo")   # authoritative, strong
tts = xai.TTS(voice="rex")   # confident, clear
tts = xai.TTS(voice="sal")   # smooth, balanced

# Custom output format
tts = xai.TTS(
    voice="rex",
    codec="mp3",
    sample_rate=44100,
    bit_rate=192000,
)

# Explicit API key (otherwise reads XAI_API_KEY env var)
tts = xai.TTS(api_key="xai-your-key-here")
```

### Configuration

| Parameter     | Type   | Default   | Description                                                           |
|---------------|--------|-----------|-----------------------------------------------------------------------|
| `api_key`     | str    | env var   | xAI API key. Falls back to `XAI_API_KEY` environment variable.        |
| `voice`       | str    | `"eve"`   | Voice ID: `"eve"`, `"ara"`, `"leo"`, `"rex"`, or `"sal"`.            |
| `language`    | str    | `"en"`    | BCP-47 language code or `"auto"` for detection.                       |
| `codec`       | str    | `"pcm"`   | Output codec: `"pcm"`, `"mp3"`, `"wav"`, `"mulaw"`, `"alaw"`.       |
| `sample_rate` | int    | `24000`   | Sample rate: `8000`–`48000` Hz.                                       |
| `bit_rate`    | int    | `None`    | MP3 bit rate (only used with `codec="mp3"`).                          |
| `base_url`    | str    | `None`    | Override the xAI TTS API endpoint.                                    |
| `session`     | object | `None`    | Optional pre-existing `aiohttp.ClientSession`.                        |

### Voices

| Voice | Tone                     | Best For                                       |
|-------|--------------------------|------------------------------------------------|
| `eve` | Energetic, upbeat        | Demos, announcements, upbeat content (default) |
| `ara` | Warm, friendly           | Conversational interfaces, hospitality         |
| `leo` | Authoritative, strong    | Instructional, educational, healthcare         |
| `rex` | Confident, clear         | Business, corporate, customer support          |
| `sal` | Smooth, balanced         | Versatile — works for any context              |

### Speech tags

Add expressiveness to synthesized speech with inline and wrapping tags:

**Inline tags** (placed where the expression should occur):
- Pauses: `[pause]` `[long-pause]` `[hum-tune]`
- Laughter: `[laugh]` `[chuckle]` `[giggle]` `[cry]`
- Mouth sounds: `[tsk]` `[tongue-click]` `[lip-smack]`
- Breathing: `[breath]` `[inhale]` `[exhale]` `[sigh]`

**Wrapping tags** (wrap text to change delivery):
- Volume: `<soft>text</soft>` `<loud>text</loud>` `<shout>text</shout>`
- Pitch/speed: `<high-pitch>text</high-pitch>` `<low-pitch>text</low-pitch>` `<slow>text</slow>` `<fast>text</fast>`
- Style: `<whisper>text</whisper>` `<sing>text</sing>`

### MP3 output

MP3 decoding requires `pydub`. Install it via the `mp3` extra:

```bash
uv add "vision-agents-plugins-xai[mp3]"
```

## Requirements

- Python 3.10+
- `xai-sdk`
- `vision-agents-core`
- Optional: `pydub` (for MP3 decoding via the `mp3` extra)

## License

Apache-2.0
