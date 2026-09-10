"""Parse Roboflow prediction payloads."""

import time
from typing import Optional

import numpy as np
import supervision as sv

from .events import DetectedObject


def detections_from_predictions(
    payload: object,
    classes: Optional[list[str]] = None,
) -> tuple[sv.Detections, dict[int, str], list[DetectedObject]]:
    """Turn a Roboflow ``predictions`` payload into detections and labelled objects.

    Accepts the inference response ``{"predictions": [...]}``, the workflow shape
    that nests it once more, or ``None``.

    Args:
        payload: What the model or workflow returned.
        classes: Keep only these labels, or every label when empty.
    """
    predictions = _prediction_dicts(payload)
    if not predictions:
        return sv.Detections.empty(), {}, []

    boxes: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    class_ids: list[int] = []
    labels: dict[int, str] = {}
    objects: list[DetectedObject] = []

    for detection in predictions:
        class_name = str(detection.get("class", ""))
        if classes and class_name not in classes:
            continue
        class_id = int(_as_float(detection.get("class_id")))
        labels[class_id] = class_name or str(class_id)
        half_width = _as_float(detection.get("width")) / 2
        half_height = _as_float(detection.get("height")) / 2
        x = _as_float(detection.get("x"))
        y = _as_float(detection.get("y"))
        x1, y1 = int(x - half_width), int(y - half_height)
        x2, y2 = int(x + half_width), int(y + half_height)
        boxes.append((x1, y1, x2, y2))
        confidences.append(_as_float(detection.get("confidence")))
        class_ids.append(class_id)
        objects.append(
            DetectedObject(label=labels[class_id], x1=x1, y1=y1, x2=x2, y2=y2)
        )

    if not class_ids:
        return sv.Detections.empty(), labels, []

    detections = sv.Detections(
        xyxy=np.array(boxes),
        confidence=np.array(confidences),
        class_id=np.array(class_ids),
    )
    return detections, labels, objects


def detection_state(
    objects: list[DetectedObject],
    confidences: Optional[list[float]] = None,
) -> dict[str, object]:
    """A compact snapshot of the latest detections for the LLM tool."""
    counts: dict[str, int] = {}
    snapshot: list[dict[str, object]] = []
    for i, obj in enumerate(objects):
        counts[obj["label"]] = counts.get(obj["label"], 0) + 1
        item: dict[str, object] = dict(obj)
        if confidences is not None and i < len(confidences):
            item["confidence"] = confidences[i]
        snapshot.append(item)
    return {"objects": snapshot, "counts": counts, "updated_at": time.time()}


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _prediction_dicts(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        payload = payload.get("predictions")
    if isinstance(payload, dict):
        payload = payload.get("predictions")
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]
