# Wizper STT Example

This directory contains an example demonstrating how to use the Wizper STT plugin with Vision Agents.

## Overview

Wizper is a speech-to-text service provided by [FAL.ai](https://fal.ai) with built-in translation capabilities.

## Features

- **Wizper STT**: Real-time speech-to-text with translation support
- **GetStream**: Real-time communication infrastructure
- **OpenAI LLM**: Intelligent response generation

## Setup

1. Install dependencies:

```bash
cd plugins/wizper/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
# Required for Wizper STT
FAL_KEY=your_fal_api_key

# Required for GetStream
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# Required for OpenAI LLM
OPENAI_API_KEY=your_openai_api_key
```

## Running the Example

```bash
uv run wizper_example.py run
```

The agent will:

1. Connect to the GetStream edge network
2. Initialize Wizper STT
3. Join a call and transcribe (and optionally translate) speech

## Customization

### Translation

```python
# Translate to French
stt = wizper.STT(target_language="fr")

# Translate to Spanish
stt = wizper.STT(target_language="es")

# Transcribe only (no translation)
stt = wizper.STT()
```

### Task Mode

```python
# Transcribe (default)
stt = wizper.STT(task="transcribe")

# Translate
stt = wizper.STT(task="translate", target_language="es")
```

## Architecture

```
User Speech
    ↓
Wizper STT (FAL.ai cloud)
    ↓
Transcript Event
    ↓
Handler logs/processes transcript
```

## Additional Resources

- [FAL.ai Documentation](https://fal.ai/docs)
- [Vision Agents Documentation](https://visionagents.ai)
- [GetStream Documentation](https://getstream.io)

## Troubleshooting

### No transcriptions

- Verify your `FAL_KEY` is valid
- Check that GetStream connection is established
- Ensure audio is being captured properly

### Translation issues

- Use valid ISO-639-1 language codes (e.g., "fr", "es", "de")
- Check FAL.ai supported languages
