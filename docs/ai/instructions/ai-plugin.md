
# Plugin Development Guide

## Example Plugin

An example plugin is located in `plugins/example`. Copying the example is the best way to create a new plugin. After copying the example be sure to:

- Update the folder name "example" to your plugin's name
- Open `pyproject.toml` and update the name, description etc
- Update the event types in your `events.py` file
- Register your events in the plugin's `__init__` method

## Folder Structure

Every plugin should follow this structure, an example for the plugin named elevenlabs:

```
/plugins/elevenlabs
- pyproject.toml
- README.md
- py.typed
- tests
- example
- vision_agents/plugins/
  - elevenlabs/
    - __init__.py
    - tts.py
    - events.py
    - tests/
```

And the logic for the plugin should live in `/plugins/elevenlabs/vision_agents/plugins/...`

## Naming

When a plugin is imported it's used like:

```python
from vision_agents.plugins import elevenlabs, anthropic

tts = elevenlabs.TTS()
llm = anthropic.LLM()
```

## Guidelines

When building the plugin read these guides:

- **TTS**: [ai-tts.md](ai-tts.md)
- **STT**: [ai-stt.md](ai-stt.md)  
- **STS/realtime/LLM**: [ai-llm.md](ai-llm.md) or [ai-realtime-llm.md](ai-realtime-llm.md)

## Update pyproject.toml

Be sure to update `pyproject.toml` at the root of this project. Add the new plugin to:

```toml
[tool.uv.sources]
myplugin = { path = "plugins/myplugin", develop = true }

[tool.uv.workspace]
members = [
    "agents-core",
    "plugins/myplugin",
    # ... other plugins
]
```