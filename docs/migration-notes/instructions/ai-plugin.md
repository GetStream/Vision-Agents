# Guide: Adding a New Stream-py Plugin

This document explains the **minimum** you need to know to create a new plugin for the `stream-py` code-base. Follow it step-by-step and you will end up with a **self-contained, tested, and discoverable** plugin.

---

## 1. Where plugins live

```
getstream/plugins/
├── stt/   # speech-to-text plugins
├── tts/   # text-to-speech plugins
└── vad/   # voice activity detection plugins
```

Create a sub-directory named after your plugin inside the correct type folder. Example for a speech-to-text (STT) plugin called `whisper`:

```
getstream/plugins/stt/whisper/
```

Every plugin is an **independent Python package** – it owns its own `pyproject.toml`, dependencies and tests.

---

## 2. Choose the correct base class

| Plugin type                    | Inherit from                |
|--------------------------------|-----------------------------|
| STT (speech-to-text)           | `getstream.plugins.stt.STT` |
| TTS (text-to-speech)           | `getstream.plugins.tts.TTS` |
| VAD (voice activity detection) | `getstream.plugins.vad.VAD` |

Import the base class and implement the abstract methods:

```python
# getstream/plugins/stt/whisper/whisper.py
from getstream.plugins.stt import STT
from typing import Optional, Dict, Any, List, Tuple
from getstream.video.rtc.track_util import PcmData

class Whisper(STT):
    """OpenAI Whisper based STT implementation."""

    def __init__(self, model: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        # Initialize your model here

    async def _process_audio_impl(
        self, pcm_data: PcmData, user_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[bool, str, Dict[str, Any]]]]:
        """
        Process audio for transcription.

        Choose your implementation mode:

        SYNCHRONOUS MODE (like Moonshine):
        - Process audio and return results
        - Base class will emit events for you

        ASYNCHRONOUS MODE (like Deepgram):
        - Emit events directly using self._emit_transcript_event()
        - Return None

        This example shows synchronous mode:
        """
        if self._is_closed:
            return None

        # Your transcription logic here
        text = self.transcribe_audio(pcm_data.samples)

        if text:
            metadata = {
                "confidence": 0.95,
                "model": self.model,
                "processing_time_ms": 100,
            }
            # Return results for base class to emit
            return [(True, text, metadata)]

        return None

    async def close(self):
        """Clean up resources."""
        self._is_closed = True

---

## 3. STT Dual-Mode Contract

STT implementations can choose between two operation modes:

### Synchronous Mode (Recommended for Offline Models)
- **Process audio immediately** and return results
- **Base class emits events** for you
- **Simple to implement** - just return `List[Tuple[bool, str, Dict]]`
- **Example**: Moonshine, Whisper, local models

```python
async def _process_audio_impl(self, pcm_data, user_metadata=None):
    # Process audio immediately
    text = await self.transcribe(pcm_data.samples)

    if text:
        metadata = {"confidence": 0.95, "processing_time_ms": 100}
        # Return results - base class will emit events
        return [(True, text, metadata)]  # is_final=True

    return None  # No results
```

### Asynchronous Mode (Recommended for Streaming Services)
- **Emit events directly** when they arrive
- **Return None** to indicate asynchronous operation
- **Real-time responsiveness** for streaming scenarios
- **Example**: Deepgram, Google Speech, Azure Speech

```python
async def _process_audio_impl(self, pcm_data, user_metadata=None):
    # Send audio to streaming service
    await self.send_audio(pcm_data.samples)

    # Events will be emitted when transcripts arrive via callbacks:
    # self._emit_transcript_event(text, user_metadata, metadata)
    # self._emit_partial_transcript_event(text, user_metadata, metadata)

    # Return None for asynchronous mode
    return None
```

### Important Rules
- **Never do both**: Don't emit events AND return results - it causes duplicates
- **Choose one mode per call**: Be consistent in your implementation
- **Use helpers**: `self._emit_transcript_event()`, `self._emit_partial_transcript_event()`, `self._emit_error_event()`

---

## 4. Shared audio utilities

For common audio processing tasks, use the shared utilities in `getstream.audio.utils`:

```python
from getstream.audio.utils import resample_audio

# High-quality resampling with automatic fallbacks
resampled_audio = resample_audio(audio_data, from_sample_rate, to_sample_rate)
```

**Available utilities:**
- `resample_audio(frame, from_sr, to_sr)` – Multi-backend resampler that prefers scipy.signal.resample_poly, falls back to torchaudio, and has a simple averaging fallback for tests

This avoids duplicating resampling logic across plugins and ensures consistent, high-quality audio processing.

---

## 5. Package skeleton

Inside your plugin folder:

```
whisper/
├── __init__.py          # re-export your class for easy import
├── whisper.py           # main implementation
├── pyproject.toml       # package metadata & deps
└── tests/
    └── test_whisper.py  # unit tests
```

`__init__.py` should expose the plugin class so the registry can import it:

```python
from .whisper import Whisper  # noqa: F401  (re-export for plugin discovery)
```

### Sample `pyproject.toml`

```toml
[project]
name = "getstream-whisper"
version = "0.1.0"
description = "Whisper STT plugin for stream-py"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}

dependencies = [
    "getstream[webrtc]",  # Use webrtc extra for audio/video/WebRTC plugins
    "openai-whisper>=0.1.0",  # replace with your specific runtime deps
]

# Optional: mark this as a plugin for dynamic discovery
[project.entry-points."getstream.plugins.stt"]
whisper = "getstream.plugins.stt.whisper:Whisper"

# Required for UV workspace
[tool.uv.sources]
getstream = { workspace = true }
getstream-plugins-stt-deepgram = { workspace = true }
getstream-plugins-tts-elevenlabs = { workspace = true }
getstream-plugins-vad-silero = { workspace = true }
```

> 📝 **Dependency Note**: Since most plugins work with audio, video, or WebRTC functionality, they should depend on `getstream[webrtc]` rather than just `getstream`. This ensures all necessary WebRTC-related dependencies are available. The `[tool.uv.sources]` section is required for the workspace setup to use the local development version of getstream.

---

## 6. Writing tests

1. Put your tests in `whisper/tests/` (or `tests/` inside the plugin folder).
2. **Reuse shared fixtures & helpers** from `getstream.plugins.test_utils`:

```python
# whisper/tests/test_whisper.py
import pytest

from getstream.plugins.test_utils import get_audio_asset
from getstream.plugins.stt.whisper import Whisper
from getstream.video.rtc.track_util import PcmData

@pytest.fixture()
def plugin():
    return Whisper()


@pytest.mark.asyncio
async def test_transcribe_simple(plugin):
    """Test basic transcription functionality."""
    # Get test audio
    audio_path = get_audio_asset("mia.mp3")

    # Create PCM data (you may need to load and convert the audio)
    pcm_data = PcmData(samples=audio_samples, sample_rate=16000, format="s16")

    # Test synchronous mode - should return results
    results = await plugin._process_audio_impl(pcm_data)

    assert results is not None, "Should return results in synchronous mode"
    assert len(results) > 0, "Should have at least one result"

    is_final, text, metadata = results[0]
    assert is_final, "Result should be final"
    assert "hello" in text.lower(), "Should contain expected text"
    assert "confidence" in metadata, "Should include confidence"


@pytest.mark.asyncio
async def test_transcript_events(plugin):
    """Test that events are emitted correctly."""
    transcript_events = []

    @plugin.on("transcript")
    def on_transcript(text, user_metadata, metadata):
        transcript_events.append((text, user_metadata, metadata))

    # Process audio using the public interface
    pcm_data = PcmData(samples=audio_samples, sample_rate=16000, format="s16")
    await plugin.process_audio(pcm_data, {"user_id": "test"})

    # Should have received transcript event
    assert len(transcript_events) == 1
    text, user_metadata, metadata = transcript_events[0]
    assert user_metadata["user_id"] == "test"
```

Running the test from project root will automatically resolve `test_utils` and load `.env` files thanks to the helper module.

```bash
uv run pytest getstream/plugins/stt/whisper/tests
```

---

## 7. Registering your plugin (optional)

If you rely on dynamic plugin discovery, expose an **entry-point** in `pyproject.toml` (see above). Otherwise, make sure your plugin is imported somewhere in client code so the class is registered.

---

## 8. Providing an example project (highly recommended)

A tiny runnable demo makes it much easier for others to try the plugin.

• Create a separate **Python package** inside the top-level `examples/` folder, e.g.:

```
examples/
└── whisper_demo/
    ├── __init__.py
    ├── main.py          # entry-point that shows how to wire the plugin
    ├── pyproject.toml   # depends on stream-py **and** your plugin package
    └── README.md        # how to run the demo
```

• The example's `pyproject.toml` should list your plugin in its dependencies so a simple `uv pip install -e .` inside the example folder pulls everything needed.

**Example dependencies:**
```toml
dependencies = [
    "getstream[webrtc]",
    "getstream-plugins-stt-whisper",  # Use the plugin package, not openai-whisper directly
    "python-dotenv>=1.1.1",
]
```

This approach ensures that:
- The example uses the exact plugin implementation you built
- All plugin dependencies are automatically included
- The example works with the workspace development setup

• Keep the demo minimal: a short script that loads some sample data (use assets from `tests/assets` if possible), runs the plugin, and prints or plays the result.

• **Optimize for readability**: The `main.py` file should be clean and easy to follow. Keep the main function focused on the core logic flow. Move error handling (try/except blocks) to the entry point (`if __name__ == "__main__":`) rather than cluttering the main function. This makes it easier for users to understand the essential steps and adapt the code to their needs.

• Examples are **optional but strongly encouraged**; they double as integration smoke-tests and real-world documentation.

---

## 9. Checklist before opening a PR

- [ ] Sub-directory under the correct plugin type.
- [ ] Inherits from the right base class.
- [ ] Contains `pyproject.toml` with precise dependencies.
- [ ] Includes at least one passing test that uses `getstream.plugins.test_utils`.
- [ ] Exports the plugin class in `__init__.py`.
- [ ] (Recommended) Adds a minimal example project under `examples/` that demonstrates basic usage.
- [ ] `uv run pytest` passes from project root.

---

## 10. Working with the UV Workspace

This project uses UV workspace configuration for seamless local development across plugins, the main package, and examples.

### Workspace Setup

The root `pyproject.toml` defines a workspace that includes:
- All plugins: `getstream/plugins/stt/*`, `getstream/plugins/tts/*`, `getstream/plugins/vad/*`
- All examples: `examples/*`
- The main `getstream` package (root)

### Development Workflow

**Initial setup:**
```bash
# From project root - installs everything in editable mode
uv sync --dev --all-extras --all-packages
```

**Adding a new plugin to the workspace:**
New plugins and examples are automatically included! The workspace uses wildcard patterns:
```toml
[tool.uv.workspace]
members = [
    "getstream/plugins/*/*",        # Automatically includes all plugins
]
exclude = [
    "**/__pycache__",              # Exclude cache directories
]
```

Just create your plugin or example with a `pyproject.toml` file in the right location and it will be automatically included.

**Plugin development:**
```bash
# All packages are automatically available in editable mode
# Changes to getstream core are immediately available to plugins
# Changes to plugins are immediately available to examples

# Run tests for a specific plugin
uv run pytest getstream/plugins/stt/whisper/tests/ -v

# Run tests for all plugins
uv run pytest getstream/plugins/ -v

# Run example
cd examples/your_example_name
uv run python main.py
```

**Key Benefits:**
- ✅ **No manual editable installs**: Everything is automatically linked
- ✅ **Immediate changes**: Modifications to any package are instantly available
- ✅ **Consistent dependencies**: Single lock file manages all versions
- ✅ **Easy testing**: Run tests across the entire workspace
- ✅ **Simple CI**: Single command to test everything

### Plugin Dependencies

In your plugin's `pyproject.toml`, you can simply depend on `getstream[webrtc]` - the workspace ensures it uses the local development version. You should also include workspace sources for all plugins to enable cross-plugin dependencies:

```toml
dependencies = [
    "getstream[webrtc]",  # Uses local workspace version during development
    "your-plugin-specific-deps>=1.0.0",
]

[tool.uv.sources]
getstream = { workspace = true }
getstream-plugins-stt-deepgram = { workspace = true }
getstream-plugins-tts-elevenlabs = { workspace = true }
getstream-plugins-vad-silero = { workspace = true }
# Add any new plugins here as they're created
```

### Publishing vs Development

- **Development**: Workspace automatically uses local versions
- **Publishing**: When published to PyPI, plugins will use the published `getstream[webrtc]` package

This gives you the best of both worlds: easy local development with proper published dependencies.

---

## 11. Checklist before opening a PR

- [ ] Sub-directory under the correct plugin type.
- [ ] Inherits from the right base class.
- [ ] Contains `pyproject.toml` with precise dependencies.
- [ ] Includes at least one passing test that uses `getstream.plugins.test_utils`.
- [ ] Exports the plugin class in `__init__.py`.
- [ ] (Recommended) Adds a minimal example project under `examples/` that demonstrates basic usage.
- [ ] `uv run pytest` passes from project root.
- [ ] When adding external dependencies, make sure to specify conservative version constraints (eg. dont use >=2.4.0)
