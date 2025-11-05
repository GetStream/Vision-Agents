"""
Roboflow plugin for vision-agents.

Provides object detection using Roboflow's hosted inference API.
"""

from .roboflow_processor import RoboflowDetectionProcessor, RoboflowVideoTrack

__all__ = [
    "RoboflowDetectionProcessor",
    "RoboflowVideoTrack",
]

