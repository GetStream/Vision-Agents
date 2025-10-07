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
pip install vision-agents-plugins-xai
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


## Requirements

- Python 3.10+
- `xai-sdk`
- `vision-agents-core`

## License

Apache-2.0
