from dataclasses import dataclass, field
from typing import TypedDict

from vision_agents.core.events import VideoProcessorDetectionEvent


class DetectedObject(TypedDict):
    """An object detected by a video processor."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


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
