"""
Stream Agents Processors Package

This package contains various processors for handling audio, video, and image processing
in Stream Agents applications.
"""

from .base_processor import (
    Processor,
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

__all__ = [
    "Processor",
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
