# Grok TTS Examples

This directory contains examples demonstrating how to use the Grok TTS plugin with Vision Agents. Each example showcases a different use case with a voice selected to match the persona.

## Examples

| Example                            | File                                | Voice | Persona                      |
|------------------------------------|-------------------------------------|-------|------------------------------|
| Basic                              | `basic_example.py`                  | Eve   | Friendly AI assistant        |
| Restaurant Host                    | `restaurant_host_example.py`        | Ara   | Upscale Italian restaurant   |
| Medical Receptionist               | `medical_receptionist_example.py`   | Sal   | Family practice front desk   |
| Customer Support                   | `customer_support_example.py`       | Rex   | SaaS product support agent   |
| Real Estate Agent                  | `real_estate_agent_example.py`      | Eve   | Property sales agent         |
| Healthcare Information             | `healthcare_example.py`             | Leo   | Telehealth wellness guide    |
| Hotel Concierge                    | `hotel_concierge_example.py`        | Ara   | Luxury hotel concierge       |

## Setup

1. Install dependencies:

```bash
cd plugins/grok_tts/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
# Required for Grok TTS
XAI_API_KEY=your_xai_api_key

# Required for speech-to-text
DEEPGRAM_API_KEY=your_deepgram_api_key

# Required for LLM
GOOGLE_API_KEY=your_google_api_key

# Required for real-time transport
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

## Running the Examples

Each example follows the same pattern — pick any one:

```bash
# Basic assistant
uv run basic_example.py run

# Restaurant host
uv run restaurant_host_example.py run

# Medical receptionist
uv run medical_receptionist_example.py run

# Customer support
uv run customer_support_example.py run

# Real estate agent
uv run real_estate_agent_example.py run

# Healthcare information
uv run healthcare_example.py run

# Hotel concierge
uv run hotel_concierge_example.py run
```

## Voice Selection Guide

Each example uses a voice that matches its persona:

- **Eve** (energetic, upbeat) — Great default for demos and enthusiastic roles like real estate
- **Ara** (warm, friendly) — Perfect for hospitality: restaurant hosts, hotel concierges
- **Leo** (authoritative, strong) — Ideal for healthcare and instructional content
- **Rex** (confident, clear) — Best for professional roles: support agents, business
- **Sal** (smooth, balanced) — Versatile choice for calm, reassuring roles like medical reception

## Customization

You can easily swap voices or adjust settings in any example:

```python
# Change voice
tts=grok_tts.TTS(voice="leo")

# Change language
tts=grok_tts.TTS(voice="ara", language="es-ES")

# Use MP3 output
tts=grok_tts.TTS(voice="eve", codec="mp3", sample_rate=44100, bit_rate=192000)
```

## Additional Resources

- [xAI TTS Documentation](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech)
- [xAI Voice API](https://x.ai/api/voice)
- [Vision Agents Documentation](https://visionagents.ai)
- [Vision Agents Plugin Guide](https://visionagents.ai/integrations/create-your-own-plugin)
