# Cartesia

[Cartesia](https://cartesia.ai) is a service that provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. It's designed for real-time voice applications, making it ideal for voice AI agents, transcription pipelines, and conversational interfaces.

The Cartesia plugin for the Stream Python AI SDK allows you to add STT and TTS functionality to your project.

## Installation

Install the Stream Cartesia plugin with

```bash
uv add "vision-agents[cartesia]"
# or directly
uv add vision-agents-plugins-cartesia
```

## Examples

Read on for some key details and check out our [Cartesia examples](https://github.com/GetStream/Vision-Agents/tree/main/plugins/cartesia/example) to see working code samples:

- in [main.py](https://github.com/GetStream/Vision-Agents/blob/main/plugins/cartesia/example/main.py) we see a voice bot that uses Cartesia STT and TTS in a Stream call
- in [narrator-example.py](https://github.com/GetStream/Vision-Agents/blob/main/plugins/cartesia/example/narrator-example.py) we see a well-prompted combination of an STT -> LLM -> TTS flow that leverages Cartesia's Ink and Sonic models to narrate a creative story from the user's input

## Initialisation

The Cartesia plugin for Stream exposes `STT` and `TTS` classes:

```python

from vision_agents.plugins import cartesia

stt = cartesia.STT()
tts = cartesia.TTS()
```

<Warning>
  To initialise without passing in the API key, make sure the `CARTESIA_API_KEY` is available as an environment variable.
  You can do this either by defining it in a `.env` file or exporting it directly in your terminal.
</Warning>

## Parameters

These are the parameters available in the Cartesia STT plugin for you to customise:

| Name                | Type            | Default                                | Description                                                                                                   |
|---------------------|-----------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `api_key`           | `str` or `None` | `None`                                 | Your Cartesia API key. If not provided, the plugin will look for the `CARTESIA_API_KEY` environment variable. |
| `model`             | `str`           | `"ink-2"`                              | ID of the Cartesia STT model to use.                                                                          |
| `sample_rate`       | `int`           | `16000`                                | Sample rate (in Hz) sent to Cartesia.                                                                         |
| `encoding`          | `str`           | `"pcm_s16le"`                          | PCM encoding sent to Cartesia.                                                                                |
| `cartesia_version`  | `str`           | `"2026-03-01"`                         | Cartesia API version used for the turn-detection websocket.                                                   |

These are the parameters available in the Cartesia TTS plugin for you to customise:

| Name          | Type            | Default                                  | Description                                                                                                   |
|---------------|-----------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `api_key`     | `str` or `None` | `None`                                   | Your Cartesia API key. If not provided, the plugin will look for the `CARTESIA_API_KEY` environment variable. |
| `model_id`    | `str`           | `"sonic-3.5"`                            | ID of the Cartesia TTS model to use.                                                                          |
| `voice_id`    | `str` or `None` | `"f9836c6e-a0bd-460e-9d3c-f7299fa60f94"` | ID of the voice to use for TTS responses.                                                                     |
| `sample_rate` | `int`           | `16000`                                  | Sample rate (in Hz) used for audio processing.                                                                |

## Functionality

### Send audio to transcribe speech

`STT` streams PCM audio to Cartesia Ink and emits transcript and turn events that Vision Agents can use for interruption and eager turn handling.

```python  theme={null}
agent = Agent(
    ...,
    stt=cartesia.STT(),
    tts=cartesia.TTS(),
)
```

### Send text to convert to speech

The `send_iter()` method sends the text passed in for the service to synthesize
and yields `TTSOutputChunk`s containing the produced PCM audio.

```python  theme={null}
async for chunk in tts.send_iter("Demo text you want AI voice to say"):
    pass
```
