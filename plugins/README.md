# GetStream Plugins

This package contains various plugins for the GetStream SDK, organized by functionality. The project uses UV workspace configuration for seamless development across plugins, the main package, and examples.

## Directory Structure

```
getstream/plugins/
├── stt/                # Speech-to-Text functionality
│   ├── deepgram/       # Deepgram STT implementation
│   └── ...             # Other STT implementations
├── tts/                # Text-to-Speech functionality
│   ├── elevenlabs/     # ElevenLabs TTS implementation
│   └── ...             # Other TTS implementations
└── vad/                # Voice Activity Detection
    ├── silero/         # Silero VAD implementation
    └── ...             # Other VAD implementations
```

## Workspace Development

All plugins are part of the UV workspace, which means:
- **Automatic editable installs**: No need to manually install plugins during development
- **Live updates**: Changes to the main `getstream` package are immediately available to plugins
- **Consistent dependencies**: Single lock file manages versions across all plugins
- **Easy testing**: Run tests for all plugins from the project root

### Development Setup

From the project root:
```bash
uv sync --all-extras --dev  # Sets up everything in editable mode
```

### Running Plugin Tests

```bash
# Run specific plugin tests
uv run pytest getstream/plugins/stt/deepgram/tests/

# Run all plugin tests
uv run pytest getstream/plugins/*/tests/

# Run tests from project root (recommended)
uv run pytest
```

## Available Plugins

### Speech-to-Text (STT)

- **Deepgram**: High-quality speech-to-text using the Deepgram API
  ```python
  from getstream.plugins import DeepgramSTT

  stt = DeepgramSTT(api_key="your_deepgram_api_key")
  ```

### Text-to-Speech (TTS)

- **ElevenLabs**: High-quality text-to-speech using the ElevenLabs API
  ```python
  from getstream.plugins import ElevenLabsTTS

  tts = ElevenLabsTTS(api_key="your_elevenlabs_api_key")
  ```

### Voice Activity Detection (VAD)

- **Silero**: Fast and accurate voice activity detection
  ```python
  from getstream.plugins import SileroVAD

  vad = SileroVAD()
  ```

## Using Plugins

All plugins follow a consistent API within their category. You can access the base classes directly:

```python
from getstream.plugins import STT, TTS, VAD
```

## Creating New Plugins

See the [Plugin Development Guide](../../ai/instructions/ai-plugin.md) for detailed instructions on creating new plugins. The workspace setup makes plugin development straightforward:

1. Create your plugin directory under the appropriate type folder
2. Add a `pyproject.toml` with `getstream[webrtc]` as a dependency
3. The workspace automatically makes it available for development
4. Run tests from the project root

For more detailed usage instructions, see the documentation for each specific plugin.
