"""
Moondream plugin for vision-agents.

This plugin provides Moondream 3 vision capabilities including object detection,
visual question answering, counting, and captioning.
"""

from .moondream_processor import (
    MoondreamProcessor,
    MoondreamVideoTrack,
)

# Re-export under the new namespace for convenience
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = [
    "MoondreamProcessor",
    "MoondreamVideoTrack",
]

