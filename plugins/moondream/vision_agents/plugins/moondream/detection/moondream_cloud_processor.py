import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import aiortc
import av
import cv2
import moondream as md
import numpy as np
from PIL import Image
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.plugins.moondream.detection.moondream_video_track import (
    MoondreamVideoTrack,
)
from vision_agents.plugins.moondream.moondream_utils import (
    annotate_detections,
    parse_detection_bbox,
)

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class CloudDetectionProcessor(VideoProcessorPublisher):
    """Performs real-time object detection on video streams using Moondream Cloud API.

    By default, the Moondream Cloud API has a 2 RPS (requests per second) rate limit,
    which can be increased by contacting the Moondream team. If you are deploying
    to your own infrastructure, consider using LocalDetectionProcessor instead.

    Args:
        api_key: API key for Moondream Cloud API. If not provided, will attempt to read
                from MOONDREAM_API_KEY environment variable.
        conf_threshold: Confidence threshold for detections (default: 0.3)
        detect_objects: Object(s) to detect. Moondream uses zero-shot detection,
                       so any object string works. Examples: "person", "car",
                       "basketball", ["person", "car", "dog"]. Default: "person"
        fps: Frame processing rate (default: 30)
        max_workers: Number of worker threads for CPU-intensive operations (default: 10)
    """

    name = "moondream_cloud"

    def __init__(
        self,
        api_key: Optional[str] = None,
        conf_threshold: float = 0.3,
        detect_objects: Union[str, List[str]] = "person",
        fps: int = 30,
        max_workers: int = 10,
    ):
        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY")
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.max_workers = max_workers
        self._shutdown = False

        # Initialize state tracking attributes
        self._last_results: Dict[str, Any] = {}
        self._last_frame_time: Optional[float] = None
        self._last_frame_pil: Optional[Image.Image] = None

        # Font configuration constants for drawing efficiency
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 2
        self._bbox_color = (0, 255, 0)
        self._text_color = (0, 0, 0)

        if not detect_objects:
            raise ValueError("detect_objects must not be empty")
        # Normalize detect_objects to list of strings
        if isinstance(detect_objects, str):
            self.detect_objects = [detect_objects]
        elif isinstance(detect_objects, list):
            if not all(isinstance(obj, str) for obj in detect_objects):
                raise ValueError("detect_objects must be str or list of strings")
            self.detect_objects = detect_objects
        else:
            raise ValueError("detect_objects must be str or list of strings")

        # Thread pool for CPU-intensive inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="moondream_processor"
        )

        # Video track for publishing (if used as video publisher)
        self._video_track: MoondreamVideoTrack = MoondreamVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # Initialize model
        self._load_model()

        logger.info("🌙 Moondream Processor initialized")
        logger.info(f"🎯 Detection configured for objects: {self.detect_objects}")

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ):
        """
        Process incoming video track.

        This method sets up the video processing pipeline:
        1. Uses shared VideoForwarder if provided, otherwise creates own
        2. Starts event consumer that calls _process_and_add_frame for each frame
        3. Frames are processed, annotated, and published via the video track
        """
        logger.info("✅ Moondream process_video starting")

        if shared_forwarder is not None:
            # Use the shared forwarder
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Moondream subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            self._video_forwarder.add_frame_handler(
                self._process_and_add_frame, fps=float(self.fps), name="moondream"
            )
        else:
            # Create our own VideoForwarder
            self._video_forwarder = VideoForwarder(
                incoming_track,  # type: ignore[arg-type]
                max_buffer=30,  # 1 second at 30fps
                fps=self.fps,
                name="moondream_forwarder",
            )

            # Add frame handler (starts automatically)
            self._video_forwarder.add_frame_handler(self._process_and_add_frame)

        logger.info("✅ Moondream video processing pipeline started")

    def publish_video_track(self):
        logger.info("📹 publish_video_track called")
        return self._video_track

    def _load_model(self):
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("api_key is required for Moondream Cloud API")

            # Initialize cloud model
            self.model = md.vl(api_key=self.api_key)
            logger.info("✅ Moondream SDK initialized")

        except Exception as e:
            logger.exception(f"❌ Failed to load Moondream model: {e}")
            raise

    async def _run_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        try:
            # Call SDK for each object type
            # The SDK's detect() is synchronous, so wrap in executor
            loop = asyncio.get_event_loop()
            all_detections = await loop.run_in_executor(
                self.executor, self._run_detection_sync, frame_array
            )

            return {"detections": all_detections}
        except Exception as e:
            logger.exception(f"❌ Cloud inference failed: {e}")
            return {"detections": []}

    def _run_detection_sync(self, frame_array: np.ndarray) -> List[Dict]:
        image = Image.fromarray(frame_array)

        if self._shutdown:
            return []

        all_detections = []

        # Call SDK for each object type
        for object_type in self.detect_objects:
            logger.debug(f"🔍 Detecting '{object_type}' via Moondream SDK")
            try:
                # Call SDK's detect method
                result = self.model.detect(image, object_type)
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect '{object_type}': {e}")
                continue

            # Parse SDK response format
            # SDK returns: {"objects": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}, ...]}
            for obj in result.get("objects", []):
                detection = parse_detection_bbox(obj, object_type, self.conf_threshold)
                if detection:
                    all_detections.append(detection)

        logger.debug(
            f"🔍 SDK returned {len(all_detections)} objects across {len(self.detect_objects)} types"
        )
        return all_detections

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        try:
            frame_array = frame.to_ndarray(format="rgb24")

            results = await self._run_inference(frame_array)

            self._last_results = results
            self._last_frame_time = asyncio.get_event_loop().time()
            self._last_frame_pil = Image.fromarray(frame_array)

            # Annotate frame with detections
            if results.get("detections"):
                frame_array = annotate_detections(
                    frame_array,
                    results,
                    font=self._font,
                    font_scale=self._font_scale,
                    font_thickness=self._font_thickness,
                    bbox_color=self._bbox_color,
                    text_color=self._text_color,
                )

            # Convert back to av.VideoFrame and publish
            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)

    async def close(self):
        """Clean up resources."""
        self._shutdown = True
        self.executor.shutdown(wait=False)
        logger.info("🛑 Moondream Processor closed")
