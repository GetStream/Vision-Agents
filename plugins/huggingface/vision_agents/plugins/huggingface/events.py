from dataclasses import dataclass, field
from typing import Any, Optional

from vision_agents.core.events import (
    DetectedObject,
    PluginBaseEvent,
    VideoProcessorDetectionEvent,
)


@dataclass
class HuggingFaceStreamEvent(PluginBaseEvent):
    """Event emitted when HuggingFace provides a stream event."""

    type: str = field(default="plugin.huggingface.stream", init=False)
    event_type: Optional[str] = None
    event_data: Optional[Any] = None


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when an LLM encounters an error."""

    type: str = field(default="plugin.llm.error", init=False)
    error_message: Optional[str] = None
    event_data: Optional[Any] = None


@dataclass
class DetectionCompletedEvent(VideoProcessorDetectionEvent):
    """Event emitted when Transformers object detection completes.

    Attributes:
        objects: Detected objects with labels and bounding boxes.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
    """

    objects: list[DetectedObject] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    type: str = field(default="plugin.huggingface.detection_completed", init=False)

    def __post_init__(self):
        self.detection_count = len(self.objects)
