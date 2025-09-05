# GetStream Common Plugin

This package provides shared abstract base classes and utilities for all GetStream AI plugins.

## Purpose

The common plugin defines the standard interfaces that all GetStream AI plugins implement:

- **STT** (Speech-to-Text): Base class for transcription plugins
- **TTS** (Text-to-Speech): Base class for synthesis plugins
- **VAD** (Voice Activity Detection): Base class for speech detection plugins
- **STS** (Speech-to-Speech): Base class for direct speech transformation plugins

## Installation

This package is automatically installed as a dependency when you install any GetStream AI plugin:

```bash
# Installing any plugin will include the common interfaces
pip install getstream-plugins-deepgram
pip install getstream-plugins-elevenlabs
# etc.
```

## Usage

Plugin developers should inherit from these base classes:

```python
from getstream.plugins.common import STT, TTS, VAD, STS

class MySTTPlugin(STT):
    async def process_audio(self, pcm_data):
        # Your STT implementation
        pass

class MyTTSPlugin(TTS):
    async def synthesize(self, text):
        # Your TTS implementation
        pass
```

## Requirements

- Python 3.10+
- getstream[webrtc]

## For Plugin Developers

If you're creating a new GetStream AI plugin, make sure to:

1. Add `getstream-plugins-common` as a dependency in your `pyproject.toml`
2. Inherit from the appropriate base class (STT, TTS, VAD, or STS)
3. Implement the required abstract methods
4. Follow the event emission patterns defined in the base classes

This ensures consistency across all GetStream AI plugins and provides users with a unified interface.
