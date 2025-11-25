# Cartesia

[Cartesia](https://cartesia.ai) is a service that provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. It's designed for real-time voice applications, making it ideal for voice AI agents, transcription pipelines, and conversational interfaces.

The Cartesia plugin for the Stream Python AI SDK allows you to add TTS functionality to your project.

## Installation

Install the Stream Cartesia plugin with

```sh  theme={null}
uv add vision-agents[cartesia]
```

## Examples

Read on for some key details and check out our [Cartesia examples](https://github.com/GetStream/vision-agents/tree/main/examples/other_examples/plugins_examples/tts_cartesia) to see working code samples:

- in [tts.py](https://github.com/GetStream/vision-agents/tree/main/examples/other_examples/plugins_examples/tts_cartesia/tts.py) we see a simple bot greeting users upon joining a call
- in [narrator-example.py](https://github.com/GetStream/vision-agents/tree/main/examples/other_examples/plugins_examples/tts_cartesia/narrator-example.py) we see a well-prompted combination of a STT -> LLM -> TTS flow that leverages the powers of Cartesia's Sonic 3 model to narrate a creative story from the user's input



## Initialisation

The Cartesia plugin for Stream exists in the form of the `TTS` class:

```python

from vision_agents.plugins import cartesia

tts = cartesia.TTS()
```

<Warning>
  To initialise without passing in the API key, make sure the `CARTESIA_API_KEY` is available as an environment variable.
  You can do this either by defining it in a `.env` file or exporting it directly in your terminal.
</Warning>

## Parameters

These are the parameters available in the CartesiaTTS plugin for you to customise:

| Name          | Type            | Default                                  | Description                                                                                                   |
| ------------- | --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `api_key`     | `str` or `None` | `None`                                   | Your Cartesia API key. If not provided, the plugin will look for the `CARTESIA_API_KEY` environment variable. |
| `model_id`    | `str`           | `"sonic-3"`                              | ID of the Cartesia STT or TTS model to use. Defaults to the recently released Sonic-3                         |
| `voice_id`    | `str` or `None` | `"f9836c6e-a0bd-460e-9d3c-f7299fa60f94"` | ID of the voice to use for TTS responses.                                                                     |
| `sample_rate` | `int`           | `16000`                                  | Sample rate (in Hz) used for audio processing.                                                                |

## Functionality

### Send text to convert to speech

The `send()` method sends the text passed in for the service to synthesize.
The resulting audio is then played through the configured output track.

```python  theme={null}
tts.send("Demo text you want AI voice to say")
```