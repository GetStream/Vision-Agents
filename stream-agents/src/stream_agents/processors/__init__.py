"""
Stream Agents Processors Package

This package contains various processors for handling audio, video, and image processing
in Stream Agents applications.
"""

from .base_processor import (
    BaseProcessor,
    AudioVideoProcessor,
    AudioProcessorMixin,
    VideoProcessorMixin,
    ImageProcessorMixin,
    VideoPublisherMixin,
    AudioPublisherMixin,
    ProcessorType,
    filter_processors,
    AudioLogger,
    ImageCapture,
)

# Optional processors with external dependencies
try:
    from .yolo_pose_processor import YOLOPoseProcessor

    __all__ = [
        "BaseProcessor",
        "AudioVideoProcessor",
        "AudioProcessorMixin",
        "VideoProcessorMixin",
        "ImageProcessorMixin",
        "VideoPublisherMixin",
        "AudioPublisherMixin",
        "ProcessorType",
        "filter_processors",
        "AudioLogger",
        "ImageCapture",
        "YOLOPoseProcessor",
    ]
except ImportError:
    # YOLOPoseProcessor requires ultralytics
    __all__ = [
        "BaseProcessor",
        "AudioVideoProcessor",
        "AudioProcessorMixin",
        "VideoProcessorMixin",
        "ImageProcessorMixin",
        "VideoPublisherMixin",
        "AudioPublisherMixin",
        "ProcessorType",
        "filter_processors",
        "AudioLogger",
        "ImageCapture",
    ]
