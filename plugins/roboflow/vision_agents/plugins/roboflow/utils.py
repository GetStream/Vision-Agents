from typing import Iterable, Optional

import cv2
import numpy as np
import supervision as sv


def annotate_image(
    image: np.ndarray,
    detections: sv.Detections,
    classes: dict[int, str],
    dim_factor: Optional[float] = None,
    text_scale: float = 0.75,
    text_padding: int = 1,
    box_thickness: int = 2,
    text_position: sv.Position = sv.Position.TOP_CENTER,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame.
    """

    # Dim the background to make detected objects brigther
    if dim_factor:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        image[mask == 0] = (image[mask == 0] * dim_factor).astype(np.uint8)

    boxed_image = sv.BoxAnnotator(thickness=box_thickness).annotate(
        image.copy(), detections
    )
    detected_class_ids: Iterable[int] = (
        detections.class_id if detections.class_id is not None else []
    )
    labels = [classes[class_id] for class_id in detected_class_ids]
    labeled_image = sv.LabelAnnotator(
        text_position=text_position,
        text_scale=text_scale,
        text_padding=text_padding,
    ).annotate(boxed_image, detections, labels)
    return labeled_image
