"""
Ultralytics plugin for vision-agents.

This plugin provides YOLO-based pose detection capabilities using the Ultralytics YOLO library.
"""

from .yolo_pose_processor import YOLOPoseProcessor


__all__ = [
    "YOLOPoseProcessor",
]
