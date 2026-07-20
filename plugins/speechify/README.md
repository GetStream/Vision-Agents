# Speechify

[Speechify](https://speechify.com) is a service that provides high quality, low latency Text-to-Speech (TTS). It's a great fit for real-time voice applications such as voice AI agents and conversational interfaces.

The Speechify plugin for the Stream Python AI SDK allows you to add TTS functionality to your project.

## Installation

Install the Stream Speechify plugin with

```bash
uv add "vision-agents[speechify]"
# or directly
uv add vision-agents-plugins-speechify
```

## Examples

Read on for some key details and check out our [Speechify example](https://github.com/GetStream/Vision-Agents/tree/main/plugins/speechify/example) to see working code:

- in [main.py](https://github.com/GetStream/Vision-Agents/blob/main/plugins/speechify/example/main.py) we see a voice bot that uses Speechify TTS in a Stream call

## Initialisation

The Speechify plugin for Stream exposes a `TTS` class:

```python
from vision_agents.plugins import speechify

tts = speechify.TTS()
```

<Warning>
  To initialise without passing in the API key, make sure the `SPEECHIFY_API_KEY` is available as an environment variable.
  You can do this either by defining it in a `.env` file or exporting it directly in your terminal.
</Warning>

## Parameters

These are the parameters available in the Speechify TTS plugin for you to customise:

| Name       | Type            | Default         | Description                                                                                                     |
|------------|-----------------|-----------------|----------------------------------------------------------------------------------------------------------------|
| `api_key`  | `str` or `None` | `None`          | Your Speechify API key. If not provided, the plugin will look for the `SPEECHIFY_API_KEY` environment variable. |
| `voice_id` | `str`           | `"geffen_32"`   | ID of the voice to use for TTS responses. See the [voices endpoint](https://docs.speechify.ai/) for options.    |
| `model`    | `str`           | `"simba-3.2"`   | ID of the Speechify model to use.                                                                              |
| `language` | `str` or `None` | `None`          | Optional ISO 639-1 + ISO 3166-1 language tag (e.g. `en-US`). When omitted the language is auto-detected.        |

## Functionality

### Send text to convert to speech

The `send_iter()` method sends the text passed in for the service to synthesize
and yields `TTSOutputChunk`s containing the produced PCM audio.

```python
async for chunk in tts.send_iter("Demo text you want AI voice to say"):
    pass
```
