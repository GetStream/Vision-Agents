# Gandr

[Gandr](https://gandr.ai) is a Text-to-Speech (TTS) API with an OpenAI compatible speech endpoint. It ships 6 voices covering 23 languages, and every render is watermarked. First audio byte in 146 ms over the open internet, 116 ms p50 first audio, server side warm.

The Gandr plugin for the Stream Python AI SDK allows you to add TTS functionality to your project.

## Installation

Install the Stream Gandr plugin with

```bash
uv add "vision-agents[gandr]"
# or directly
uv add vision-agents-plugins-gandr
```

Keys are available at [gandr.ai](https://gandr.ai). The free tier is 50,000 tokens.

## Initialisation

The Gandr plugin exposes a `TTS` class:

```python
from vision_agents.plugins import gandr

tts = gandr.TTS()
```

<Warning>
  To initialise without passing in the API key, make sure the `GANDR_API_KEY` is available as an environment variable.
  You can do this either by defining it in a `.env` file or exporting it directly in your terminal.
</Warning>

## Parameters

These are the parameters available in the Gandr TTS plugin for you to customise:

| Name       | Type            | Default                     | Description                                                                                             |
|------------|-----------------|-----------------------------|---------------------------------------------------------------------------------------------------------|
| `api_key`  | `str` or `None` | `None`                      | Your Gandr API key. If not provided, the plugin will look for the `GANDR_API_KEY` environment variable. |
| `model`    | `str`           | `"tts-1"`                   | Model name sent to the speech endpoint.                                                                 |
| `voice`    | `str`           | `"gandr-mia"`               | Which Gandr voice to use.                                                                               |
| `base_url` | `str`           | `"https://tts.gandr.ai/v1"` | Gandr API base URL.                                                                                     |

Available voices: `gandr-mia`, `gandr-ava`, `gandr-jenny`, `gandr-dane`, `gandr-leo`, `gandr-lewis`.

## Functionality

### Send text to convert to speech

The `send_iter()` method sends the text passed in for the service to synthesize
and yields `TTSOutputChunk`s containing the produced PCM audio.

```python
async for chunk in tts.send_iter("Demo text you want AI voice to say"):
    pass
```

The plugin requests `pcm` output: headerless signed 16 bit little endian mono at 24000 Hz, so the bytes are wrapped directly with no decode step. The API accepts at most 2000 characters per request.

Use it in an agent like any other TTS plugin:

```python
agent = Agent(
    ...,
    tts=gandr.TTS(),
)
```
