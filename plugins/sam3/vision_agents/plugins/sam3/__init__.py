"""
SAM 3 plugin for vision-agents.

This plugin provides SAM 3 (Segment Anything with Concepts) capabilities for
real-time video segmentation with text-based prompting.
"""

from vision_agents.plugins.sam3.processor import VideoSegmentationProcessor
from vision_agents.plugins.sam3.video_track import Sam3VideoTrack

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = [
    "VideoSegmentationProcessor",
    "Sam3VideoTrack",
]

