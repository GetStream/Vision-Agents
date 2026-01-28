"""
Annotation Processor for visualizing LLM-generated annotations on video.

This processor subscribes to LLM response events, parses visual annotations
(bounding boxes, circles, polygons, etc.) from the response text, and draws
them on video frames. This allows LLMs to "point to" or "circle" objects in
the video feed.

Example usage:
    from vision_agents.core.processors import AnnotationProcessor

    processor = AnnotationProcessor(
        fps=30,
        annotation_duration=10.0,  # Show annotations for 10 seconds
        default_style={'color': (0, 255, 0), 'thickness': 3}
    )

    agent = Agent(
        processors=[processor],
        llm=llm,
        # ... other params
    )

Coordinate Format (Gemini):
    Gemini returns bounding boxes in [y_min, x_min, y_max, x_max] format
    with coordinates normalized to 0-1000 (not pixels). The processor
    automatically converts these to absolute pixel coordinates.

When the LLM responds with JSON like:
    [{"box_2d": [91, 122, 433, 458], "label": "face"}]

The processor will:
1. Parse the coordinates (y_min=91, x_min=122, y_max=433, x_max=458)
2. Convert from 0-1000 scale to actual frame pixels
3. Draw a green rectangle at the correct position
4. Add the label "face" above the box
5. Display this for 10 seconds
"""

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import aiortc
import av
import cv2
import numpy as np

from vision_agents.core.events.manager import EventManager
from vision_agents.core.llm.events import LLMResponseCompletedEvent, VLMAnnotationEvent
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

from .annotation_types import (
    Annotation,
    AnnotationStyle,
    BoundingBox,
    Circle,
    Line,
    Point,
    Polygon,
    TextLabel,
)

logger = logging.getLogger(__name__)


class AnnotationProcessor(VideoProcessorPublisher):
    """
    Processes and visualizes LLM-generated visual annotations on video frames.

    This processor:
    - Subscribes to LLM response events
    - Parses JSON annotations from LLM responses
    - Draws annotations (boxes, circles, polygons, etc.) on video frames
    - Publishes annotated video as the agent's video output
    - Auto-clears annotations after configurable duration

    Args:
        fps: Frame processing rate (default: 30)
        annotation_duration: How long to display annotations in seconds (default: 10.0)
        default_style: Default visual style for annotations (color, thickness, etc.)
        max_workers: Number of worker threads for frame processing (default: 4)
        auto_normalize: If True, treats coordinates as normalized (0-1) (default: False)
    """

    name = "annotation_processor"

    def __init__(
        self,
        fps: int = 30,
        annotation_duration: float = 10.0,
        default_style: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        auto_normalize: bool = False,
    ):
        self.fps = fps
        self.annotation_duration = annotation_duration
        self.auto_normalize = auto_normalize
        self.max_workers = max_workers

        # Default visual style
        if default_style is None:
            default_style = {
                "color": (0, 255, 0),  # Green in BGR
                "thickness": 3,
                "font_scale": 0.6,
                "font_thickness": 2,
                "fill": False,
            }
        self.default_style = AnnotationStyle(**default_style)

        # Current annotations to display
        self._annotations: List[Annotation] = []
        self._annotations_lock = asyncio.Lock()
        self._annotations_expire_time: float = 0.0

        # Reference frame - the exact frame that was analyzed by the VLM
        # When set, annotations are drawn on this frame instead of live video
        self._reference_frame: Optional[av.VideoFrame] = None

        # Video components - create track with same FPS as processor
        self._video_track: QueuedVideoTrack = QueuedVideoTrack(fps=self.fps)
        self._video_forwarder: Optional[VideoForwarder] = None

        # Thread pool for CPU-intensive drawing
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="annotation_processor"
        )

        # Shutdown flag
        self._shutdown = False

        # Event manager (will be attached by agent)
        self.events: Optional[EventManager] = None

        logger.info("📐 Annotation Processor initialized")

    def attach_agent(self, agent: Any) -> None:
        """
        Called by agent to attach event manager.

        Args:
            agent: The agent instance
        """
        self.events = agent.events

        # Subscribe to LLM response events
        # Note: We need to wrap the method call because subscribe expects a function
        # with only event parameters (no self)
        if self.events:

            @self.events.subscribe
            async def on_llm_response(event: LLMResponseCompletedEvent) -> None:
                await self._on_llm_response_completed(event)

            @self.events.subscribe
            async def on_vlm_annotation(event: VLMAnnotationEvent) -> None:
                await self._on_vlm_annotation(event)

            logger.info("✅ AnnotationProcessor subscribed to LLM response events")

    async def _on_llm_response_completed(
        self, event: LLMResponseCompletedEvent
    ) -> None:
        """
        Handle LLM response completion events.

        Parses annotations from the response text and updates the display.

        Args:
            event: The LLM response completed event
        """
        if not event.text:
            return

        try:
            # Parse annotations from response text
            annotations = self._parse_annotations(event.text)

            if annotations:
                async with self._annotations_lock:
                    self._annotations = annotations
                    self._annotations_expire_time = (
                        time.time() + self.annotation_duration
                    )
                logger.info(
                    f"📐 Parsed {len(annotations)} annotation(s) from LLM response"
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse annotations from LLM response: {e}")

    async def _on_vlm_annotation(self, event: VLMAnnotationEvent) -> None:
        """
        Handle VLM annotation events with reference frame.

        This is the preferred path for annotations - it includes the exact frame
        that was analyzed by the VLM, ensuring annotations are drawn correctly.

        Args:
            event: The VLM annotation event with reference frame
        """
        if not event.annotations_json:
            return

        try:
            # Parse annotations from the JSON
            annotations = self._parse_annotations(event.annotations_json)

            if annotations:
                async with self._annotations_lock:
                    self._annotations = annotations
                    self._annotations_expire_time = (
                        time.time() + self.annotation_duration
                    )
                    # Store the reference frame - this is the key difference!
                    # Annotations will be drawn on this exact frame
                    if event.reference_frame is not None:
                        self._reference_frame = event.reference_frame
                        logger.info(
                            f"📐 Stored reference frame for {len(annotations)} annotation(s)"
                        )
                    else:
                        logger.info(
                            f"📐 Parsed {len(annotations)} annotation(s) (no reference frame)"
                        )
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse annotations from VLM event: {e}")

    def _parse_annotations(self, text: str) -> List[Annotation]:
        """
        Parse annotations from LLM response text.

        Looks for JSON arrays in the format:
        [
            {"box_2d": [y_min, x_min, y_max, x_max], "label": "face"},
            {"circle": {"center": [y, x], "radius": 50}, "label": "head"},
            ...
        ]

        Also supports markdown code blocks: ```json\n...\n```

        Args:
            text: The LLM response text

        Returns:
            List of parsed Annotation objects
        """
        annotations = []

        # Try to extract JSON from markdown code blocks first
        markdown_pattern = r"```json\s*([\s\S]*?)\s*```"
        markdown_matches = re.findall(markdown_pattern, text)

        json_strings = markdown_matches

        # Also try to find inline JSON arrays
        inline_pattern = r"\[\s*\{[^\]]+\]\s*"
        inline_matches = re.findall(inline_pattern, text, re.DOTALL)
        json_strings.extend(inline_matches)

        for json_str in json_strings:
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    for item in data:
                        annotation = self._parse_annotation_item(item)
                        if annotation:
                            annotations.append(annotation)
            except json.JSONDecodeError:
                continue

        return annotations

    def _parse_annotation_item(self, item: Dict[str, Any]) -> Optional[Annotation]:
        """
        Parse a single annotation item from JSON.

        Supports Gemini's native format where box_2d is [y_min, x_min, y_max, x_max]
        with coordinates normalized to 0-1000.

        Args:
            item: Dictionary containing annotation data

        Returns:
            Parsed Annotation object or None if invalid
        """
        try:
            # Parse bounding box: {"box_2d": [y_min, x_min, y_max, x_max], "label": "..."}
            # Gemini returns coordinates normalized to 0-1000
            # IMPORTANT: Gemini format is [y_min, x_min, y_max, x_max] - y comes FIRST
            if "box_2d" in item:
                coords = item["box_2d"]
                if len(coords) == 4:
                    # Gemini format: [y_min, x_min, y_max, x_max] normalized 0-1000
                    y_min, x_min, y_max, x_max = coords
                    return BoundingBox(
                        x1=x_min / 1000.0,
                        y1=y_min / 1000.0,
                        x2=x_max / 1000.0,
                        y2=y_max / 1000.0,
                        label=item.get("label"),
                        confidence=item.get("confidence"),
                        style=self.default_style,
                        normalized=True,  # Gemini coordinates are always normalized
                        format="xyxy",
                    )

            # Parse bounding box (alternative format): {"bbox": [x, y, w, h]}
            if "bbox" in item:
                coords = item["bbox"]
                if len(coords) == 4:
                    return BoundingBox(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        label=item.get("label"),
                        confidence=item.get("confidence"),
                        style=self.default_style,
                        normalized=self.auto_normalize,
                        format="xywh",
                    )

            # Parse circle: {"circle": {"center": [y, x], "radius": r}, "label": "..."}
            # Gemini returns center in [y, x] order, coordinates normalized to 0-1000
            if "circle" in item:
                circle_data = item["circle"]
                center = circle_data.get("center", [])
                radius = circle_data.get("radius")
                if len(center) == 2 and radius:
                    # Gemini format: center is [y, x] normalized 0-1000
                    center_y, center_x = center
                    return Circle(
                        center_x=center_x / 1000.0,
                        center_y=center_y / 1000.0,
                        radius=radius / 1000.0,  # Radius also normalized
                        label=item.get("label"),
                        style=self.default_style,
                        normalized=True,  # Gemini coordinates are always normalized
                    )

            # Parse polygon: {"polygon": [[y1,x1], [y2,x2], ...], "label": "..."}
            # Gemini returns points in [y, x] order, coordinates normalized to 0-1000
            if "polygon" in item:
                points = item["polygon"]
                if isinstance(points, list) and len(points) >= 3:
                    # Convert from Gemini's [y, x] to [x, y] and normalize from 0-1000 to 0-1
                    points_tuples = [
                        (p[1] / 1000.0, p[0] / 1000.0) for p in points if len(p) == 2
                    ]
                    if points_tuples:
                        return Polygon(
                            points=points_tuples,
                            label=item.get("label"),
                            style=self.default_style,
                            normalized=True,  # Gemini coordinates are always normalized
                        )

            # Parse point: {"point": [y, x], "label": "..."}
            # Gemini returns point in [y, x] order, coordinates normalized to 0-1000
            if "point" in item:
                point = item["point"]
                if len(point) == 2:
                    # Gemini format: [y, x] normalized 0-1000
                    point_y, point_x = point
                    return Point(
                        x=point_x / 1000.0,
                        y=point_y / 1000.0,
                        label=item.get("label"),
                        marker_size=item.get("marker_size", 10),
                        style=self.default_style,
                        normalized=True,  # Gemini coordinates are always normalized
                    )

            # Parse line: {"line": {"start": [y1, x1], "end": [y2, x2]}, "label": "..."}
            # Gemini returns points in [y, x] order, coordinates normalized to 0-1000
            if "line" in item:
                line_data = item["line"]
                start = line_data.get("start", [])
                end = line_data.get("end", [])
                if len(start) == 2 and len(end) == 2:
                    # Gemini format: [y, x] normalized 0-1000
                    start_y, start_x = start
                    end_y, end_x = end
                    return Line(
                        x1=start_x / 1000.0,
                        y1=start_y / 1000.0,
                        x2=end_x / 1000.0,
                        y2=end_y / 1000.0,
                        label=item.get("label"),
                        style=self.default_style,
                        normalized=True,  # Gemini coordinates are always normalized
                    )

            # Parse text label: {"label": {"text": "...", "position": [y, x]}}
            # Gemini returns position in [y, x] order, coordinates normalized to 0-1000
            if "label" in item and isinstance(item["label"], dict):
                label_data = item["label"]
                text = label_data.get("text")
                position = label_data.get("position", [])
                if text and len(position) == 2:
                    # Gemini format: [y, x] normalized 0-1000
                    pos_y, pos_x = position
                    return TextLabel(
                        text=text,
                        x=pos_x / 1000.0,
                        y=pos_y / 1000.0,
                        style=self.default_style,
                        normalized=True,  # Gemini coordinates are always normalized
                    )

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.debug(f"Failed to parse annotation item: {e}")
            return None

        return None

    def _draw_annotations_sync(
        self, frame: np.ndarray, annotations: List[Annotation]
    ) -> np.ndarray:
        """
        Draw annotations on a frame (synchronous, CPU-bound).

        Args:
            frame: Frame in RGB format (numpy array)
            annotations: List of annotations to draw

        Returns:
            Frame with annotations drawn
        """
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]

        for annotation in annotations:
            # Convert normalized coordinates to absolute if needed
            if annotation.normalized:
                if isinstance(annotation, BoundingBox):
                    annotation = annotation.to_absolute(width, height)
                elif isinstance(annotation, Circle):
                    annotation = annotation.to_absolute(width, height)
                elif isinstance(annotation, Polygon):
                    annotation = annotation.to_absolute(width, height)
                elif isinstance(annotation, Point):
                    annotation = annotation.to_absolute(width, height)
                elif isinstance(annotation, Line):
                    annotation = annotation.to_absolute(width, height)
                elif isinstance(annotation, TextLabel):
                    annotation = annotation.to_absolute(width, height)

            style = annotation.style or self.default_style

            # Draw based on annotation type
            if isinstance(annotation, BoundingBox):
                x1, y1, x2, y2 = annotation.get_corners()
                cv2.rectangle(
                    frame_bgr, (x1, y1), (x2, y2), style.color, style.thickness
                )
                if annotation.label:
                    label_text = annotation.label
                    if annotation.confidence:
                        label_text += f" {annotation.confidence:.2f}"
                    cv2.putText(
                        frame_bgr,
                        label_text,
                        (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        style.font_scale,
                        style.color,
                        style.font_thickness,
                    )

            elif isinstance(annotation, Circle):
                center = (int(annotation.center_x), int(annotation.center_y))
                radius = int(annotation.radius)
                cv2.circle(
                    frame_bgr,
                    center,
                    radius,
                    style.color,
                    style.thickness if not style.fill else -1,
                )
                if annotation.label:
                    cv2.putText(
                        frame_bgr,
                        annotation.label,
                        (center[0] - radius, max(10, center[1] - radius - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        style.font_scale,
                        style.color,
                        style.font_thickness,
                    )

            elif isinstance(annotation, Polygon):
                points = np.array(annotation.points, dtype=np.int32)
                cv2.polylines(
                    frame_bgr, [points], True, style.color, style.thickness
                )
                if annotation.label and len(annotation.points) > 0:
                    label_pos = annotation.points[0]
                    cv2.putText(
                        frame_bgr,
                        annotation.label,
                        (int(label_pos[0]), max(10, int(label_pos[1]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        style.font_scale,
                        style.color,
                        style.font_thickness,
                    )

            elif isinstance(annotation, Point):
                point = (int(annotation.x), int(annotation.y))
                cv2.circle(
                    frame_bgr,
                    point,
                    annotation.marker_size,
                    style.color,
                    -1,  # Filled
                )
                if annotation.label:
                    cv2.putText(
                        frame_bgr,
                        annotation.label,
                        (point[0] + 15, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        style.font_scale,
                        style.color,
                        style.font_thickness,
                    )

            elif isinstance(annotation, Line):
                start = (int(annotation.x1), int(annotation.y1))
                end = (int(annotation.x2), int(annotation.y2))
                cv2.line(frame_bgr, start, end, style.color, style.thickness)
                if annotation.label:
                    mid_x = (start[0] + end[0]) // 2
                    mid_y = (start[1] + end[1]) // 2
                    cv2.putText(
                        frame_bgr,
                        annotation.label,
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        style.font_scale,
                        style.color,
                        style.font_thickness,
                    )

            elif isinstance(annotation, TextLabel):
                pos = (int(annotation.x), int(annotation.y))
                cv2.putText(
                    frame_bgr,
                    annotation.text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    style.font_scale,
                    style.color,
                    style.font_thickness,
                )

        # Convert back to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        """
        Process a frame with annotations and add to output track.

        When a reference frame is available (from VLMAnnotationEvent), annotations
        are drawn on that exact frame to ensure they match what the VLM analyzed.
        Otherwise, annotations are drawn on the current live frame.

        Args:
            frame: Input video frame
        """
        if self._shutdown:
            return

        try:
            current_time = time.time()

            # Check if annotations have expired
            async with self._annotations_lock:
                if current_time > self._annotations_expire_time:
                    if self._annotations:
                        logger.debug("⏱️ Annotations expired, clearing")
                        self._annotations = []
                        # Also clear the reference frame when annotations expire
                        self._reference_frame = None

                annotations_to_draw = self._annotations.copy()
                reference_frame = self._reference_frame

            # Determine which frame to use for drawing
            # If we have a reference frame, use it (ensures annotation accuracy)
            # Otherwise fall back to the live frame
            if annotations_to_draw and reference_frame is not None:
                frame_array = reference_frame.to_ndarray(format="rgb24")
            else:
                frame_array = frame.to_ndarray(format="rgb24")

            # Draw annotations if present
            if annotations_to_draw:
                loop = asyncio.get_event_loop()
                annotated_array = await loop.run_in_executor(
                    self._executor,
                    self._draw_annotations_sync,
                    frame_array,
                    annotations_to_draw,
                )
            else:
                annotated_array = frame_array

            # Convert back to av.VideoFrame
            processed_frame = av.VideoFrame.from_ndarray(
                annotated_array, format="rgb24"
            )

            # Add to output track
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Set up video processing pipeline.

        Args:
            track: Input video track
            participant_id: Optional participant ID
            shared_forwarder: Optional shared video forwarder
        """
        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            self._video_forwarder.add_frame_handler(
                self._process_and_add_frame, fps=float(self.fps), name=self.name
            )
        else:
            self._video_forwarder = VideoForwarder(
                track,  # type: ignore[arg-type]
                max_buffer=30,
                fps=self.fps,
                name=f"{self.name}_forwarder",
            )
            self._video_forwarder.add_frame_handler(self._process_and_add_frame)

        logger.info("✅ Annotation processor video processing started")

    async def stop_processing(self) -> None:
        """Stop processing video tracks."""
        if self._video_forwarder:
            await self._video_forwarder.remove_frame_handler(self._process_and_add_frame)
            self._video_forwarder = None
            logger.info("🛑 Stopped annotation processor video processing")

    def publish_video_track(self):
        """Return the video track for publishing."""
        return self._video_track

    async def close(self):
        """Clean up resources."""
        self._shutdown = True
        await self.stop_processing()
        self._executor.shutdown(wait=False)
        async with self._annotations_lock:
            self._annotations.clear()
            self._reference_frame = None
        logger.info("🛑 Annotation processor closed")
