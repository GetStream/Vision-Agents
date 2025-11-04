"""
Moondream plugin for vision-agents.

This plugin provides Moondream 3 vision capabilities including object detection,
visual question answering, counting, and captioning.
"""

from .moondream_cloud_processor import (
    CloudDetectionProcessor,
)
from .moondream_local_processor import (
    LocalDetectionProcessor,
)
from .moondream_video_track import (
    MoondreamVideoTrack,
)

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = [
    "CloudDetectionProcessor",
    "LocalDetectionProcessor",
    "MoondreamVideoTrack",
]

