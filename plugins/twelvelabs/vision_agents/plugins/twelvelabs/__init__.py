"""TwelveLabs plugin for vision-agents.

Provides the TwelveLabs Pegasus video-understanding model as a first-class
VideoLLM, letting an agent reason about short video clips from a live track.
"""

from .pegasus_vlm import PegasusVLM
from .pegasus_vlm import PegasusVLM as VLM

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = [
    "PegasusVLM",
    "VLM",
]
