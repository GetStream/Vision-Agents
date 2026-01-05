import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import aiortc
import av
import cv2
import numpy as np
from numpy import ndarray
from PIL import Image
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


class YOLOPoseProcessor(VideoProcessorPublisher):
    """
    Yolo pose detection processor.

    - It receives the images via process_image
    - Converts it to an ND array

    """

    name = "yolo_pose"

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        imgsz: int = 512,
        device: str = "cpu",
        max_workers: int = 24,
        fps: int = 30,
        enable_hand_tracking: bool = True,
        enable_wrist_highlights: bool = True,
    ):
        self.model_path = model_path
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.enable_hand_tracking = enable_hand_tracking
        self.enable_wrist_highlights = enable_wrist_highlights
        self._last_frame: Optional[Image.Image] = None
        self._video_forwarder: Optional[VideoForwarder] = None

        # Initialize YOLO model
        self._load_model()

        # Thread pool for CPU-intensive pose processing
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="yolo_pose_processor"
        )
        self._shutdown = False

        # Video track for publishing (if used as video publisher)
        self._video_track = QueuedVideoTrack()

        logger.info(f"🤖 YOLO Pose Processor initialized with model: {model_path}")

    def _load_model(self):
        from ultralytics import YOLO

        """Load the YOLO pose model."""
        if not Path(self.model_path).exists():
            logger.warning(
                f"Model file {self.model_path} not found. YOLO will download it automatically."
            )

        self.pose_model = YOLO(self.model_path)
        self.pose_model.to(self.device)
        logger.info(f"✅ YOLO pose model loaded: {self.model_path} on {self.device}")

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ):
        if self._video_forwarder is not None:
            logger.info(
                "🎥 Stopping the ongoing YOLO video processing because the new video track is published"
            )
            await self._video_forwarder.remove_frame_handler(
                self._add_pose_and_add_frame
            )

        logger.info(f"🎥 Starting YOLO video processing at {self.fps} FPS")
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                incoming_track,
                max_buffer=self.fps,  # 1 second
                fps=self.fps,
                name="yolo_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._add_pose_and_add_frame, fps=float(self.fps), name="yolo"
        )

    async def _add_pose_and_add_frame(self, frame: av.VideoFrame):
        # Exit early if the processor is already closing
        if self._shutdown:
            return
        frame_with_pose = await self.add_pose_to_frame(frame)
        if frame_with_pose is None:
            logger.info(
                "add_pose_to_frame did not return a frame, returning the original frame instead."
            )
            await self._video_track.add_frame(frame)
        else:
            await self._video_track.add_frame(frame_with_pose)

    async def add_pose_to_frame(self, frame: av.VideoFrame) -> Optional[av.VideoFrame]:
        try:
            frame_array = frame.to_ndarray(format="rgb24")
            array_with_pose, pose = await self.add_pose_to_ndarray(frame_array)
            frame_with_pose = av.VideoFrame.from_ndarray(array_with_pose)
            return frame_with_pose
        except Exception:
            logger.exception("add_pose_to_frame failed")
            return None

    async def add_pose_to_image(self, image: Image.Image) -> tuple[Image.Image, Any]:
        """
        Adds the pose to the given image. Note that this is slightly less efficient compared to
        using add_pose_to_ndarray directly
        """
        frame_array = np.array(image)
        array_with_pose, pose_data = await self.add_pose_to_ndarray(frame_array)
        annotated_image = Image.fromarray(array_with_pose)

        return annotated_image, pose_data

    async def add_pose_to_ndarray(
        self, frame_array: np.ndarray
    ) -> tuple[ndarray, dict[str, Any]]:
        """
        Adds the pose information to the given frame array. This is slightly faster than using add_pose_to_image
        """
        annotated_array, pose_data = await self._process_pose_async(frame_array)
        return annotated_array, pose_data

    def publish_video_track(self):
        """
        Creates a yolo pose video track
        """
        return self._video_track

    async def _process_pose_async(
        self, frame_array: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Async wrapper for pose processing.

        Args:
            frame_array: Input frame as numpy array

        Returns:
            Tuple of (annotated_frame_array, pose_data)
        """
        loop = asyncio.get_event_loop()
        frame_height, frame_width = frame_array.shape[:2]

        logger.debug(f"🤖 Starting pose processing: {frame_width}x{frame_height}")
        start_time = time.perf_counter()

        try:
            # Add timeout to prevent blocking
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, self._process_pose_sync, frame_array
                ),
                timeout=12.0,  # 12 second timeout
            )
            processing_time = time.perf_counter() - start_time
            logger.debug(
                f"✅ Pose processing completed in {processing_time:.3f}s for {frame_width}x{frame_height}"
            )
            return result
        except asyncio.TimeoutError:
            processing_time = time.perf_counter() - start_time
            logger.warning(
                f"⏰ Pose processing TIMEOUT after {processing_time:.3f}s for {frame_width}x{frame_height} - returning original frame"
            )
            return frame_array, {}
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            logger.error(
                f"❌ Error in async pose processing after {processing_time:.3f}s for {frame_width}x{frame_height}: {e}"
            )
            return frame_array, {}

    def _process_pose_sync(
        self, frame_array: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        try:
            if self._shutdown:
                logger.debug("🛑 Pose processing skipped - processor shutdown")
                return frame_array, {}

            # Store original dimensions for quality preservation
            original_height, original_width = frame_array.shape[:2]
            logger.debug(
                f"🔍 Running YOLO pose detection on {original_width}x{original_height} frame"
            )

            # Run pose detection
            yolo_start = time.perf_counter()
            pose_results = self.pose_model(
                frame_array,
                verbose=False,
                # imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
            )
            yolo_time = time.perf_counter() - yolo_start
            logger.debug(f"🎯 YOLO inference completed in {yolo_time:.3f}s")

            if not pose_results:
                logger.debug("❌ No pose results detected")
                return frame_array, {}

            # Apply pose results to current frame
            annotated_frame = frame_array.copy()
            pose_data: Dict[str, Any] = {"persons": []}

            # Process each detected person
            for person_idx, result in enumerate(pose_results):
                if not result.keypoints:
                    continue

                keypoints = result.keypoints
                if keypoints is not None and len(keypoints.data) > 0:
                    kpts = keypoints.data[0].cpu().numpy()  # Get person's keypoints

                    # Store pose data
                    person_data = {
                        "person_id": person_idx,
                        "keypoints": kpts.tolist(),
                        "confidence": float(np.mean(kpts[:, 2])),  # Average confidence
                    }
                    pose_data["persons"].append(person_data)

                    # Draw keypoints
                    for i, (x, y, conf) in enumerate(kpts):
                        if conf > self.conf_threshold:  # Only draw confident keypoints
                            cv2.circle(
                                annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1
                            )

                    # Draw skeleton connections
                    self._draw_skeleton_connections(annotated_frame, kpts)

                    # Highlight wrist positions if enabled
                    if self.enable_wrist_highlights:
                        self._highlight_wrists(annotated_frame, kpts)

            logger.debug(
                f"✅ Pose processing completed successfully - detected {len(pose_data['persons'])} persons"
            )
            return annotated_frame, pose_data

        except Exception as e:
            logger.error(f"❌ Error in pose processing: {e}")
            return frame_array, {}

    def _draw_skeleton_connections(self, annotated_frame: np.ndarray, kpts: np.ndarray):
        """
        Draw skeleton connections on the annotated frame.
        Based on the kickboxing example's connection logic.
        """
        # Basic skeleton connections
        connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head connections
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arm connections
            (5, 11),
            (6, 12),
            (11, 12),  # Torso connections
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Leg connections
        ]

        # Enhanced hand and wrist connections for detailed tracking
        if self.enable_hand_tracking:
            hand_connections = [
                # Right hand connections
                (9, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),  # Right hand thumb
                (9, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),  # Right hand index
                (9, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (28, 29),  # Right hand middle
                (9, 30),
                (30, 31),
                (31, 32),
                (32, 33),
                (33, 34),  # Right hand ring
                (9, 35),
                (35, 36),
                (36, 37),
                (37, 38),
                (38, 39),  # Right hand pinky
                # Left hand connections (if available)
                (8, 45),
                (45, 46),
                (46, 47),
                (47, 48),
                (48, 49),  # Left hand thumb
                (8, 50),
                (50, 51),
                (51, 52),
                (52, 53),
                (53, 54),  # Left hand index
                (8, 55),
                (55, 56),
                (56, 57),
                (57, 58),
                (58, 59),  # Left hand middle
                (8, 60),
                (60, 61),
                (61, 62),
                (62, 63),
                (63, 64),  # Left hand ring
                (8, 65),
                (65, 66),
                (66, 67),
                (67, 68),
                (68, 69),  # Left hand pinky
            ]
            connections.extend(hand_connections)

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(kpts) and end_idx < len(kpts):
                x1, y1, c1 = kpts[start_idx]
                x2, y2, c2 = kpts[end_idx]
                if c1 > self.conf_threshold and c2 > self.conf_threshold:
                    # Use different colors for different body parts
                    if start_idx >= 9 and start_idx <= 39:  # Right hand
                        color = (0, 255, 255)  # Cyan for right hand
                    elif start_idx >= 40 and start_idx <= 69:  # Left hand
                        color = (255, 255, 0)  # Yellow for left hand
                    else:  # Main body
                        color = (255, 0, 0)  # Blue for main skeleton
                    cv2.line(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )

    def _highlight_wrists(self, annotated_frame: np.ndarray, kpts: np.ndarray):
        """
        Highlight wrist positions with special markers.
        Based on the kickboxing example's wrist highlighting logic.
        """
        wrist_keypoints = [9, 10]  # Right and left wrists
        for wrist_idx in wrist_keypoints:
            if wrist_idx < len(kpts):
                x, y, conf = kpts[wrist_idx]
                if conf > self.conf_threshold:
                    # Draw larger, more visible wrist markers
                    cv2.circle(
                        annotated_frame, (int(x), int(y)), 8, (0, 0, 255), -1
                    )  # Red wrist markers
                    cv2.circle(
                        annotated_frame, (int(x), int(y)), 10, (255, 255, 255), 2
                    )  # White outline

                    # Add wrist labels
                    wrist_label = "R Wrist" if wrist_idx == 9 else "L Wrist"
                    cv2.putText(
                        annotated_frame,
                        wrist_label,
                        (int(x) + 15, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        wrist_label,
                        (int(x) + 15, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

    async def close(self):
        """Clean up resources."""
        self._shutdown = True
        self.executor.shutdown(wait=False)
