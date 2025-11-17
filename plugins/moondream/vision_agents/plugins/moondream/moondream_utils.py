from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import torch


def handle_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def parse_detection_bbox(
    obj: Dict, object_type: str, conf_threshold: float
) -> Optional[Dict]:
    confidence = obj.get("confidence", 1.0)

    # Filter by confidence threshold
    if confidence < conf_threshold:
        return None

    bbox = [
        obj.get("x_min", 0),
        obj.get("y_min", 0),
        obj.get("x_max", 0),
        obj.get("y_max", 0),
    ]

    return {"label": object_type, "bbox": bbox, "confidence": confidence}


def normalize_bbox_coordinates(bbox: List[float], width: int, height: int) -> tuple:
    if len(bbox) != 4:
        return (0, 0, 0, 0)

    x1, y1, x2, y2 = bbox

    # Check if normalized coordinates (between 0 and 1)
    if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
        # Convert to pixel coordinates
        return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    else:
        # Already pixel coordinates
        return int(x1), int(y1), int(x2), int(y2)


def annotate_detections(
    frame_array: np.ndarray,
    results: Dict[str, Any],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 2,
    bbox_color: tuple = (0, 255, 0),
    text_color: tuple = (0, 0, 0),
) -> np.ndarray:
    annotated = frame_array.copy()

    detections = results.get("detections", [])
    if not detections:
        return annotated

    height, width = frame_array.shape[:2]

    # Pre-calculate baseline text metrics once per frame for efficiency
    sample_text = "object 0.00"  # Representative text for baseline calculation
    (_, text_height), baseline = cv2.getTextSize(
        sample_text, font, font_scale, font_thickness
    )

    for detection in detections:
        # Parse bounding box and normalize to pixel coordinates
        bbox = detection.get("bbox", [])
        x1, y1, x2, y2 = normalize_bbox_coordinates(bbox, width, height)

        # Skip invalid bounding boxes
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue

        # Get label and confidence
        label = detection.get("label", "object")
        conf = detection.get("confidence", 0.0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 2)

        # Draw label background
        label_text = f"{label} {conf:.2f}"
        # Calculate text width for this specific label (varies by content)
        (text_width, _), _ = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            bbox_color,
            -1,
        )

        # Draw label text using cached parameters
        cv2.putText(
            annotated,
            label_text,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

    return annotated
