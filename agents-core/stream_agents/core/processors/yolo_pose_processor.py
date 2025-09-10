"""
YOLO Pose Detection Processor

This processor implements real-time pose detection using YOLO llm,
extracting the pose detection logic from the kickboxing example and
adapting it to the new processor architecture.
"""

import asyncio
import base64
import io
import logging
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from PIL import Image
from aiortc import VideoStreamTrack
import av


from .base_processor import (
    AudioVideoProcessor,
    ImageProcessorMixin,
    VideoProcessorMixin,
    VideoPublisherMixin,
)


logger = logging.getLogger(__name__)


class YOLOPoseVideoTrack(VideoStreamTrack):
    """Custom video track for YOLO pose detection output."""

    def __init__(self):
        super().__init__()
        self.frame_queue: asyncio.Queue[Image.Image] = asyncio.Queue(maxsize=10)
        self.last_frame = Image.new("RGB", (640, 480), color="black")
        self._stopped = False
        # Set video quality parameters
        self.width = 640
        self.height = 480
        logger.info(
            f"🎥 YOLOPoseVideoTrack initialized with dimensions: {self.width}x{self.height}"
        )

    async def add_frame(self, image: Image.Image):
        """Add a frame to the video track."""
        if self._stopped:
            return

        try:
            # Ensure the image is the correct size
            if image.size != (self.width, self.height):
                image = image.resize(
                    (self.width, self.height), Image.Resampling.LANCZOS
                )

            # Try to add frame without blocking if queue is full
            try:
                self.frame_queue.put_nowait(image)
            except asyncio.QueueFull:
                # Drop oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(image)
                except asyncio.QueueEmpty:
                    pass

        except Exception as e:
            logger.error(f"Error adding frame to video track: {e}")

    async def recv(self) -> av.frame.Frame:
        """Receive the next video frame."""
        if self._stopped:
            raise Exception("Track stopped")

        try:
            # Try to get a frame from queue with short timeout
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.02)
            if frame:
                self.last_frame = frame
                logger.debug(f"Got frame from queue: {frame.size}")
        except asyncio.TimeoutError:
            logger.debug("No frame in queue, using last frame")
            pass
        except Exception as e:
            logger.warning(f"Error getting frame from queue: {e}")

        # Get timestamp for the frame
        pts, time_base = await self.next_timestamp()

        # Ensure the image is the correct size before creating video frame
        if self.last_frame.size != (self.width, self.height):
            self.last_frame = self.last_frame.resize(
                (self.width, self.height), Image.Resampling.LANCZOS
            )

        # Create av.VideoFrame from PIL Image
        av_frame = av.VideoFrame.from_image(self.last_frame)
        av_frame.pts = pts
        av_frame.time_base = time_base

        logger.debug(f"Returning video frame: {av_frame.width}x{av_frame.height}")
        return av_frame

    def stop(self):
        """Stop the video track."""
        self._stopped = True
        logger.info("🛑 YOLOPoseVideoTrack stopped")


class YOLOPoseProcessor(
    AudioVideoProcessor, ImageProcessorMixin, VideoProcessorMixin, VideoPublisherMixin
):
    """
    YOLO-based pose detection processor that can:
    1. Process images with pose detection overlays
    2. Process video tracks with real-time pose detection
    3. Publish transformed video with pose annotations
    """

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        imgsz: int = 512,
        device: str = "cpu",
        max_workers: int = 2,
        interval: int = 0,
        enable_hand_tracking: bool = True,
        enable_wrist_highlights: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the YOLO Pose Processor.

        Args:
            model_path: Path to YOLO pose model file
            conf_threshold: Confidence threshold for pose detection
            imgsz: Image size for YOLO inference
            device: Device to run inference on ('cpu' or 'cuda')
            max_workers: Number of worker threads for processing
            interval: Processing interval in seconds (0 for every frame)
            enable_hand_tracking: Whether to draw detailed hand connections
            enable_wrist_highlights: Whether to highlight wrist positions
        """
        super().__init__(interval=interval, receive_audio=False, receive_video=True)

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.enable_hand_tracking = enable_hand_tracking
        self.enable_wrist_highlights = enable_wrist_highlights
        self._last_frame: Optional[Image.Image] = None

        # Initialize YOLO model
        self._load_model()

        # Thread pool for CPU-intensive pose processing
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="yolo_pose_processor"
        )
        self._shutdown = False

        # Video track for publishing (if used as video publisher)
        self._video_track: Optional[YOLOPoseVideoTrack] = None

        logger.info(f"🤖 YOLO Pose Processor initialized with model: {model_path}")

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLOPoseProcessor. "
                "Install it with: pip install ultralytics"
            )

        """Load the YOLO pose model."""
        if not Path(self.model_path).exists():
            logger.warning(
                f"Model file {self.model_path} not found. YOLO will download it automatically."
            )

        self.pose_model = YOLO(self.model_path)
        self.pose_model.to(self.device)
        logger.info(f"✅ YOLO pose model loaded: {self.model_path} on {self.device}")

    def create_video_track(self):
        """Create a video track for publishing pose-annotated frames."""

        self._video_track = YOLOPoseVideoTrack()
        logger.info("🎥 YOLO pose video track created for publishing")
        return self._video_track

    async def process_image(
        self,
        image: Image.Image,
        user_id: str,
        metadata: Optional[dict[Any, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image with pose detection.

        Args:
            image: PIL Image to process
            user_id: ID of the user
            metadata: Additional metadata

        Returns:
            Dictionary containing pose detection results and annotated image
        """
        if not self.should_process():
            return None

        try:
            # Convert PIL to numpy array
            frame_array = np.array(image)

            # Process pose detection
            annotated_array, pose_data = await self._process_pose_async(frame_array)

            # Convert back to PIL Image
            annotated_image = Image.fromarray(annotated_array)

            # Publish annotated frame to output track if available
            if self._video_track:
                self._last_frame = annotated_image
                await self._video_track.add_frame(annotated_image)
                logger.debug("🎥 Published pose-annotated frame to video track")

            logger.debug(f"🤖 Processed pose detection for user {user_id}")

            return {
                "annotated_image": annotated_image,
                "pose_data": pose_data,
                "user_id": user_id,
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"❌ Error processing image pose detection: {e}")
            return None

    async def process_video(self, track, user_id: str):
        """
        Process video frames from the input track with pose detection.
        Note: The Agent class handles frame-by-frame processing via process_image,
        so this method just logs that video processing is active.

        Args:
            track: Video track to process
            user_id: ID of the user
        """
        # The actual frame processing happens in process_image method
        # called by the Agent for each frame

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

        try:
            # Add timeout to prevent blocking
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, self._process_pose_sync, frame_array
                ),
                timeout=2.0,  # 2 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Pose processing timed out, returning original frame")
            return frame_array, {}
        except Exception as e:
            logger.error(f"Error in async pose processing: {e}")
            return frame_array, {}

    def _process_pose_sync(
        self, frame_array: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Synchronous pose processing for thread pool.
        Based on the kickboxing example's pose detection logic.

        Args:
            frame_array: Input frame as numpy array

        Returns:
            Tuple of (annotated_frame_array, pose_data)
        """
        try:
            if self._shutdown:
                return frame_array, {}

            # Store original dimensions for quality preservation
            original_height, original_width = frame_array.shape[:2]

            # Run pose detection
            pose_results = self.pose_model(
                frame_array,
                verbose=False,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
            )

            if not pose_results:
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

            return annotated_frame, pose_data

        except Exception as e:
            logger.error(f"Error in pose processing: {e}")
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

    def state(self) -> Dict[str, Any]:
        """
        Return current processor state for LLM context.

        Returns:
            Dictionary containing processor state information
        """
        return {
            "processor_type": "YOLO Pose Detection",
            "model_path": self.model_path,
            "confidence_threshold": self.conf_threshold,
            "device": self.device,
            "last_frame": self._last_frame,
            "hand_tracking_enabled": self.enable_hand_tracking,
            "wrist_highlights_enabled": self.enable_wrist_highlights,
            "processing_interval": self.interval,
            "status": "active" if not self._shutdown else "shutdown",
        }

    def input(self) -> Optional[Dict[str, Any]]:
        """Return input for OpenAI API."""
        if self._last_frame:
            buffered = io.BytesIO()
            self._last_frame.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # TODO: this should be a utility method
            # Return the official OpenAI responses.create format
            return {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_data}",
            }
        return None

    def cleanup(self):
        """Clean up resources."""
        self._shutdown = True
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
        logger.info("🧹 YOLO Pose Processor cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, "_shutdown") and not self._shutdown:
            self.cleanup()
