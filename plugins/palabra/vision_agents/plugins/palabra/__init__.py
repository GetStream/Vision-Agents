from .tts import TTS, WS_URL_EU, WS_URL_US, PalabraTTSError

# Re-export under the new namespace for convenience
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = ["TTS", "WS_URL_EU", "WS_URL_US", "PalabraTTSError"]
