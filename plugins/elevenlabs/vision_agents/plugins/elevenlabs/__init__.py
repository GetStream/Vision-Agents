from .tts import TTS
from .stt import STT

# Re-export under the new namespace for convenience
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = ["TTS", "STT"]
