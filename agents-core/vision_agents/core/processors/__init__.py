"""
Stream Agents Processors Package

This package contains various processors for handling audio, video, and image processing
in Stream Agents applications.
"""

from .annotation_processor import AnnotationProcessor
from .annotation_types import (
    Annotation,
    AnnotationStyle,
    AnnotationType,
    BoundingBox,
    Circle,
    Line,
    Point,
    Polygon,
    TextLabel,
)
from .base_processor import (
    AudioProcessorPublisher,
    AudioPublisher,
    AudioProcessor,
    Processor,
    VideoProcessorPublisher,
    VideoPublisher,
    VideoProcessor,
)

__all__ = [
    "Processor",
    "VideoPublisher",
    "AudioPublisher",
    "VideoProcessor",
    "AudioProcessor",
    "AudioProcessorPublisher",
    "VideoProcessorPublisher",
    # Annotation support
    "AnnotationProcessor",
    "Annotation",
    "AnnotationStyle",
    "AnnotationType",
    "BoundingBox",
    "Circle",
    "Line",
    "Point",
    "Polygon",
    "TextLabel",
]
