
# Plugin Development Guide

## 1. Copy the example plugin folder

A sample plugin is located in `plugins/sample_plugin`. Start by copying the sample plugin and renaming it

## 2. Update your new plugin

After you copy the example be sure to:

- Open `pyproject.toml` and update the name, description etc

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

## PCM / Audio management

Use `PcmData` and other utils available from the `getstream.video.rtc.track_util` module. Do not write code that directly manipulates PCM, use the audio utilities instead.

## Plugin README.MD Format
The README.md of the plugin should contain a standardise format which includes an intro briefly describing the functionality of the plugin and what it allows developers to build when paired with Vision Agents. See example below: 
```md 
# Plugin Name 
 "Name of plugin"

## Features 
- Few bullet points calling attention to the main functionality of the plugins

## Installation 
Installation instructions in the format of a command using uv add. As an example:
```
    uv add vision-agents[PLUGIN-NAME]
```

## Usage
Basic code snippet showing the usage in code. Example: 
```python
from vision_agents.plugins import ultralytics # Plugin import 

# Create a YOLO pose processor
processor = ultralytics.YOLOPoseProcessor(
    model_path="yolo11n-pose.pt",
    conf_threshold=0.5,
    device="cpu",
    enable_hand_tracking=True,
    enable_wrist_highlights=True
)
```

## Configuration
Parameters developers can specify to adjust the plugin behaviour. This should be shown as a table with the parameter name, a short description derived from the code docs and the accepted values (either data type or default value)

## Dependencies
Bullet point list of dependencies the plugin requires to work correctly
```