# HuggingFace Plugin for Vision Agents

HuggingFace Inference integration for Vision Agents. Supports both text-only LLM and vision language models (VLM) through HuggingFace's Inference Providers API.

## Installation

```bash
uv add vision-agents[huggingface]
```

## Configuration

Set your HuggingFace API token:

```bash
export HF_TOKEN=your_huggingface_token
```

## Usage

### Text-only LLM

```python
from vision_agents.plugins import huggingface

llm = huggingface.LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="together",  # optional: auto-selects if omitted. You can also pass "fastest" and "cheapest" here if interested in throughput-efficiency or cost-efficiency
)

response = await llm.simple_response("Hello, how are you?")
print(response.text)
```

### Vision Language Model (VLM)

```python
from vision_agents.plugins import huggingface

vlm = huggingface.VLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    fps=1,
    frame_buffer_seconds=10,
)

# VLM automatically buffers video frames when used with an Agent
response = await vlm.simple_response("What do you see?")
print(response.text)
```

### With Function Calling

```python
from vision_agents.plugins import huggingface

llm = huggingface.LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

@llm.register_function()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."

response = await llm.simple_response("What's the weather in Paris?")
```

## Supported Providers

HuggingFace's Inference Providers API supports multiple backends:

- Together AI
- Groq
- Cerebras
- Replicate
- Fireworks
- And more

Specify a provider explicitly or let HuggingFace auto-select:

```python
llm = huggingface.LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="groq",
)
```

## API Reference

### `huggingface.LLM`

Text-only language model integration.

**Parameters:**
- `model` (str): HuggingFace model ID
- `api_key` (str, optional): HuggingFace API token (defaults to `HF_TOKEN` env var)
- `provider` (str, optional): Inference provider name

### `huggingface.VLM`

Vision language model integration with video frame buffering.

**Parameters:**
- `model` (str): HuggingFace model ID
- `api_key` (str, optional): HuggingFace API token (defaults to `HF_TOKEN` env var)
- `provider` (str, optional): Inference provider name
- `fps` (int): Frames per second to buffer (default: 1)
- `frame_buffer_seconds` (int): Seconds of video to buffer (default: 10)
