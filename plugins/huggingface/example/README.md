# HuggingFace LLM Example

This example demonstrates how to use HuggingFace's Inference Providers API with Vision Agents to create a conversational
voice agent.

## Setup

1. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

- `HF_TOKEN` - Your HuggingFace API token
- `STREAM_API_KEY` - Your Stream API key
- `STREAM_API_SECRET` - Your Stream API secret
- `DEEPGRAM_API_KEY` - Your Deepgram API key

2. Install dependencies:

```bash
uv sync
```

3. Run the example:

```bash
uv run main.py run
```

## Features

- Uses HuggingFace Inference Providers API for LLM
- Supports multiple providers (Together, Groq, Cerebras, etc.)
- Uses Deepgram for speech-to-text and text-to-speech
- Integrates with Stream for real-time communication

## Models

You can customize the model by changing the `model` parameter:

```python
llm = huggingface.LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
```

You can also specify a provider:

```python
llm = huggingface.LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="together",  # or "groq", "cerebras", etc.
)
```
