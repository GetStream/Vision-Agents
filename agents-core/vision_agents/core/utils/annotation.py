"""Shared annotation utilities for video processor plugins.

Requires ``supervision`` and ``opencv-python`` at runtime. These are not
core dependencies — plugins that use annotation must declare them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    import numpy as np
    import supervision as sv


def annotate_image(
    image: np.ndarray,
    detections: sv.Detections,
    classes: dict[int, str],
    dim_factor: Optional[float] = None,
    mask_opacity: Optional[float] = None,
    text_scale: float = 0.75,
    text_padding: int = 1,
    box_thickness: int = 2,
    text_position: sv.Position | None = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, and optional mask overlays on an image.

    Args:
        image: RGB image as a numpy array (H, W, 3).
        detections: Supervision ``Detections`` object.
        classes: Mapping of class ID to label string.
        dim_factor: Dim background outside detected boxes (0–1). ``None`` to skip.
        mask_opacity: Mask overlay opacity (0–1). ``None`` to skip mask annotation.
            Only applied when ``detections.mask`` is not ``None``.
        text_scale: Label text scale.
        text_padding: Label text padding.
        box_thickness: Bounding box line thickness.
        text_position: Label position relative to box.
    """
    import cv2
    import numpy as np
    import supervision as sv

    if text_position is None:
        text_position = sv.Position.TOP_CENTER

    if dim_factor:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        image[mask == 0] = (image[mask == 0] * dim_factor).astype(np.uint8)

    annotated = image.copy()

    if mask_opacity is not None and detections.mask is not None:
        annotated = sv.MaskAnnotator(opacity=mask_opacity).annotate(
            annotated, detections
        )

    annotated = sv.BoxAnnotator(thickness=box_thickness).annotate(annotated, detections)
    detected_class_ids: Iterable[int] = (
        detections.class_id if detections.class_id is not None else []
    )
    labels = [classes[class_id] for class_id in detected_class_ids]
    annotated = sv.LabelAnnotator(
        text_position=text_position,
        text_scale=text_scale,
        text_padding=text_padding,
    ).annotate(annotated, detections, labels)
    return annotated
