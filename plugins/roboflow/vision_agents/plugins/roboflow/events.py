import typing
from dataclasses import dataclass, field
from typing import Optional
import supervision as sv
from vision_agents.core.events import PluginBaseEvent


class DetectedObject(typing.TypedDict):
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DetectionCompletedEvent(PluginBaseEvent):
    """Event emitted when object detection is completed.

    Attributes:
        objects: The objects detected in the video track.
        raw_detections: The raw `sv.Detections` returned by Roboflow inference.
        image_width: width of the source image.
        image_height: height of the source image.
        inference_time_ms: Time taken for inference in milliseconds.
        model_id: Identifier of the model used for detection.
    """

    objects: list[DetectedObject] = field(default_factory=list)
    raw_detections: sv.Detections = field(default_factory=sv.Detections.empty)
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: Optional[float] = None
    model_id: Optional[str] = None
    type: str = field(default="plugin.roboflow.detection_completed", init=False)
