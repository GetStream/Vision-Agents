
# Plugin

## Folder structure

Every plugin should follow this structure, an example for the plugin named elevenlabs

/plugins/elevenlabs
- pyproject.toml
- README.md
- py.typed
- stream_agents/plugins/

And the logic for the plugin should live in /plugins/elevenlabs/stream_agents/plugins/...

## Naming

When a plugin is imported it's used like

```
from stream_agents.plugins import elevenlabs, anthropic

tts = elevenlabs.TTS()
llm = Anthropic.LLM()
```

## Example plugin

An example plugin is located in plugins/example
Copying the example is the best way to create a new plugin
After copying the example be sure to:

- update the folder name "example" to your plugin's name
- open pyproject.toml and update the name, description etc

## Guidelines

When building the plugin read these guides:

- TTS: ai-tts.md
- STT: ai-stt.md
- STS/realtime/LLM: ai-llm.md
- Video processor: ai-video-processor.md
