from typing import Optional

import cv2
import numpy as np
import supervision as sv


def annotate_image(
    image: np.ndarray,
    detections: sv.Detections,
    classes: dict[int, str],
    dim_factor: Optional[float] = None,
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

    boxed_image = sv.BoxAnnotator(thickness=1).annotate(image.copy(), detections)
    labels = [classes[class_id] for class_id in detections.class_id]
    labeled_image = sv.LabelAnnotator(
        text_position=sv.Position.BOTTOM_CENTER,
        text_scale=0.25,
        text_padding=1,
    ).annotate(boxed_image, detections, labels)
    return labeled_image
