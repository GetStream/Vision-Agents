"""
Roboflow plugin for vision-agents.

Provides object detection using Roboflow's hosted inference API.
"""

from .events import DetectionCompletedEvent
from .roboflow_cloud_processor import RoboflowCloudDetectionProcessor
from .roboflow_local_processor import RoboflowLocalDetectionProcessor
from .utils import LocalTrack

__all__ = [
    "RoboflowLocalDetectionProcessor",
    "RoboflowCloudDetectionProcessor",
    "LocalTrack",
    "DetectionCompletedEvent",
]
