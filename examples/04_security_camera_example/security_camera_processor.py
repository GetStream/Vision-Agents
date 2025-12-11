import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiortc
import av
import cv2
import face_recognition
import numpy as np

from pathlib import Path

from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    VideoProcessorMixin,
    VideoPublisherMixin,
)
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Represents a detected face with metadata."""

    face_id: str
    face_image: np.ndarray
    face_encoding: np.ndarray
    first_seen: float
    last_seen: float
    bbox: tuple
    detection_count: int = 1


@dataclass
class PackageDetection:
    """Represents a detected package with metadata."""

    package_id: str
    package_image: np.ndarray
    first_seen: float
    last_seen: float
    bbox: tuple
    confidence: float
    detection_count: int = 1


class SecurityCameraProcessor(
    AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin
):
    """
    Security camera processor that detects and recognizes faces and packages.

    This processor:
    - Detects faces in real-time using OpenCV
    - Uses face_recognition library to identify unique individuals
    - Detects packages using YOLO object detection model
    - Prevents duplicate entries for the same person/package
    - Maintains a 30-minute sliding window of unique visitors and packages
    - Displays visitor count, package count, and thumbnails in a grid overlay

    Args:
        fps: Frame processing rate (default: 5)
        max_workers: Number of worker threads (default: 10)
        time_window: Time window in seconds to track faces/packages (default: 1800 = 30 minutes)
        thumbnail_size: Size of face/package thumbnails in overlay (default: 80)
        detection_interval: Minimum seconds between face detections (default: 2)
        face_match_tolerance: Face recognition tolerance (default: 0.6, lower = stricter)
        model_path: Path to YOLO model file (default: "yolo11n.pt")
        device: Device to run YOLO model on (default: "cpu")
        package_detection_interval: Minimum seconds between package detections (default: 3)
        package_fps: FPS for package detection (default: 1)
        package_conf_threshold: Confidence threshold for package detection (default: 0.3)
    """

    name = "security_camera"

    def __init__(
        self,
        fps: int = 5,
        max_workers: int = 10,
        time_window: int = 1800,
        thumbnail_size: int = 80,
        detection_interval: float = 2.0,
        face_match_tolerance: float = 0.6,
        model_path: str = "weights.pt",
        device: str = "cpu",
        package_detection_interval: float = 3.0,
        package_fps: int = 1,
        package_conf_threshold: float = 0.3,
    ):
        super().__init__(interval=0, receive_audio=False, receive_video=True)

        self.fps = fps
        self.max_workers = max_workers
        self.time_window = time_window
        self.thumbnail_size = thumbnail_size
        self.detection_interval = detection_interval
        self.face_match_tolerance = face_match_tolerance
        self.package_detection_interval = package_detection_interval
        self.package_fps = package_fps
        self.package_conf_threshold = package_conf_threshold

        # Storage for unique detected faces (keyed by face_id)
        self._detected_faces: Dict[str, FaceDetection] = {}
        self._last_detection_time = 0.0

        # Storage for unique detected packages (keyed by package_id)
        self._detected_packages: Dict[str, PackageDetection] = {}
        self._last_package_detection_time = 0.0

        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="security_camera"
        )

        # Video track for publishing
        self._video_track: QueuedVideoTrack = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # Load OpenCV face detector
        self._face_cascade = None

        # Initialize YOLO model for package detection
        self.model_path = model_path
        self.device = device
        self.yolo_model: Optional[Any] = None
        self.package_detect_classes = [
            "package",
            "box",
            "parcel",
            "suitcase",
            "backpack",
        ]

        logger.info("🎥 Security Camera Processor initialized")
        logger.info(f"📊 Time window: {time_window}s ({time_window // 60} minutes)")
        logger.info(f"🖼️ Thumbnail size: {thumbnail_size}x{thumbnail_size}")
        logger.info(
            f"📦 Package detection: {package_fps} FPS, interval: {package_detection_interval}s"
        )

    async def warmup(self):
        """Load OpenCV Haar Cascade face detector and YOLO model."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✅ Face detector loaded")
        except Exception as e:
            logger.exception(f"❌ Failed to load face detector: {e}")
            raise

        try:
            from ultralytics import YOLO

            loop = asyncio.get_event_loop()

            def load_yolo_model():
                if not Path(self.model_path).exists():
                    logger.warning(
                        f"Model file {self.model_path} not found. YOLO will download it automatically."
                    )
                logger.debug("Loading model file...")
                model = YOLO(self.model_path)
                model.to(self.device)
                return model

            self.yolo_model = await loop.run_in_executor(self.executor, load_yolo_model)
            logger.info(f"✅ YOLO model loaded: {self.model_path} on {self.device}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load YOLO model: {e}")
            logger.warning("⚠️ Package detection will be disabled")
            self.yolo_model = None

    def _cleanup_old_faces(self, current_time: float):
        cutoff_time = current_time - self.time_window

        # Remove faces whose last_seen is older than the cutoff
        faces_to_remove = [
            face_id
            for face_id, face in self._detected_faces.items()
            if face.last_seen < cutoff_time
        ]

        for face_id in faces_to_remove:
            del self._detected_faces[face_id]

        removed = len(faces_to_remove)
        if removed > 0:
            logger.debug(f"🧹 Cleaned up {removed} old face(s)")

    def _cleanup_old_packages(self, current_time: float):
        cutoff_time = current_time - self.time_window

        # Remove packages whose last_seen is older than the cutoff
        packages_to_remove = [
            package_id
            for package_id, package in self._detected_packages.items()
            if package.last_seen < cutoff_time
        ]

        for package_id in packages_to_remove:
            del self._detected_packages[package_id]

        removed = len(packages_to_remove)
        if removed > 0:
            logger.debug(f"🧹 Cleaned up {removed} old package(s)")

    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: (x, y, w, h) format
            bbox2: (x, y, w, h) format

        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x_min, y_min, x_max, y_max) format
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _find_matching_package(
        self, bbox: tuple, iou_threshold: float = 0.3
    ) -> Optional[str]:
        """Find matching package based on IoU overlap.

        Args:
            bbox: (x, y, w, h) format
            iou_threshold: Minimum IoU to consider a match (default: 0.3)

        Returns:
            package_id if match found, None otherwise
        """
        if not self._detected_packages:
            return None

        best_match_id = None
        best_iou = 0.0

        for package_id, package in self._detected_packages.items():
            iou = self._calculate_iou(bbox, package.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match_id = package_id

        return best_match_id

    def _detect_faces_sync(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        face_locations = face_recognition.face_locations(frame_rgb, model="hog")

        if not face_locations:
            return []

        # Generate face encodings
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        # Convert to list of dicts with bbox in (x, y, w, h) format
        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Convert from (top, right, bottom, left) to (x, y, w, h)
            x = left
            y = top
            w = right - left
            h = bottom - top

            results.append({"bbox": (x, y, w, h), "encoding": encoding})

        return results

    def _find_matching_face(self, face_encoding: np.ndarray) -> Optional[str]:
        if not self._detected_faces:
            return None

        # Get all existing face encodings
        known_face_ids = list(self._detected_faces.keys())
        known_encodings = [
            self._detected_faces[face_id].face_encoding for face_id in known_face_ids
        ]

        # Compare against all known faces
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=self.face_match_tolerance
        )

        # If we found a match, return the face_id
        for i, is_match in enumerate(matches):
            if is_match:
                return known_face_ids[i]

        return None

    async def _detect_and_store_faces(
        self, frame_bgr: np.ndarray, current_time: float
    ) -> int:
        """
        Detect faces in frame and store new unique faces or update existing ones.

        Returns:
            Number of new unique faces detected
        """
        # Check if enough time has passed since last detection
        if current_time - self._last_detection_time < self.detection_interval:
            return 0

        # Convert BGR to RGB for face_recognition library
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        detected_faces = await loop.run_in_executor(
            self.executor, self._detect_faces_sync, frame_rgb
        )

        new_faces = 0
        updated_faces = 0

        for face_data in detected_faces:
            x, y, w, h = face_data["bbox"]
            face_encoding = face_data["encoding"]

            # Extract face thumbnail (convert back to BGR for storage)
            face_roi = frame_bgr[y : y + h, x : x + w]
            face_thumbnail = cv2.resize(
                face_roi, (self.thumbnail_size, self.thumbnail_size)
            )

            # Check if this face matches any existing face
            matching_face_id = self._find_matching_face(face_encoding)

            if matching_face_id:
                # Update existing face
                face_detection = self._detected_faces[matching_face_id]
                face_detection.last_seen = current_time
                face_detection.detection_count += 1
                face_detection.bbox = (x, y, w, h)
                # Update thumbnail to latest image
                face_detection.face_image = face_thumbnail
                updated_faces += 1
                logger.debug(
                    f"🔄 Updated existing face {matching_face_id[:8]} "
                    f"(seen {face_detection.detection_count} times)"
                )
            else:
                # New unique face
                face_id = str(uuid.uuid4())
                detection = FaceDetection(
                    face_id=face_id,
                    face_image=face_thumbnail,
                    face_encoding=face_encoding,
                    first_seen=current_time,
                    last_seen=current_time,
                    bbox=(x, y, w, h),
                    detection_count=1,
                )
                self._detected_faces[face_id] = detection
                new_faces += 1
                logger.info(f"👤 New unique visitor detected: {face_id[:8]}")

        if new_faces > 0 or updated_faces > 0:
            self._last_detection_time = current_time
            logger.info(
                f"📊 Detection summary - New: {new_faces}, Updated: {updated_faces}, "
                f"Total unique visitors: {len(self._detected_faces)}"
            )

        return new_faces

    def _detect_packages_sync(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO package detection synchronously.

        Args:
            frame_rgb: Frame in RGB format

        Returns:
            List of detection dicts with bbox and confidence
        """
        if not self.yolo_model:
            return []

        height, width = frame_rgb.shape[:2]
        all_detections = []

        try:
            results = self.yolo_model(
                frame_rgb,
                verbose=False,
                conf=self.package_conf_threshold,
                device=self.device,
            )

            if not results:
                return []

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = class_names[cls_id].lower()

                if any(
                    detect_class in class_name
                    for detect_class in self.package_detect_classes
                ):
                    x_min, y_min, x_max, y_max = box

                    x_min = int(max(0, min(x_min, width - 1)))
                    y_min = int(max(0, min(y_min, height - 1)))
                    x_max = int(max(x_min + 1, min(x_max, width)))
                    y_max = int(max(y_min + 1, min(y_max, height)))

                    x = x_min
                    y = y_min
                    w = x_max - x_min
                    h = y_max - y_min

                    all_detections.append(
                        {
                            "bbox": (x, y, w, h),
                            "confidence": float(conf),
                            "label": class_name,
                        }
                    )

        except Exception as e:
            logger.warning(f"⚠️ Failed to detect packages with YOLO: {e}")

        return all_detections

    async def _detect_and_store_packages(
        self, frame_bgr: np.ndarray, current_time: float
    ) -> int:
        """
        Detect packages in frame and store new unique packages or update existing ones.

        Returns:
            Number of new unique packages detected
        """
        if not self.yolo_model:
            return 0

        # Check if enough time has passed since last detection
        if (
            current_time - self._last_package_detection_time
            < self.package_detection_interval
        ):
            return 0

        # Convert BGR to RGB for Moondream
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        detected_packages = await loop.run_in_executor(
            self.executor, self._detect_packages_sync, frame_rgb
        )

        new_packages = 0
        updated_packages = 0

        for package_data in detected_packages:
            x, y, w, h = package_data["bbox"]
            confidence = package_data["confidence"]

            # Ensure coordinates are integers and within frame bounds
            height, width = frame_bgr.shape[:2]
            x = int(max(0, min(x, width - 1)))
            y = int(max(0, min(y, height - 1)))
            w = int(max(1, min(w, width - x)))
            h = int(max(1, min(h, height - y)))

            # Extract package thumbnail
            package_roi = frame_bgr[y : y + h, x : x + w]
            if package_roi.size == 0:
                continue

            package_thumbnail = cv2.resize(
                package_roi, (self.thumbnail_size, self.thumbnail_size)
            )

            # Check if this package matches any existing package
            matching_package_id = self._find_matching_package((x, y, w, h))

            if matching_package_id:
                # Update existing package
                package_detection = self._detected_packages[matching_package_id]
                package_detection.last_seen = current_time
                package_detection.detection_count += 1
                package_detection.bbox = (x, y, w, h)
                package_detection.confidence = max(
                    package_detection.confidence, confidence
                )
                package_detection.package_image = package_thumbnail
                updated_packages += 1
                logger.debug(
                    f"🔄 Updated existing package {matching_package_id[:8]} "
                    f"(seen {package_detection.detection_count} times)"
                )
            else:
                # New unique package
                package_id = str(uuid.uuid4())
                detection = PackageDetection(
                    package_id=package_id,
                    package_image=package_thumbnail,
                    first_seen=current_time,
                    last_seen=current_time,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    detection_count=1,
                )
                self._detected_packages[package_id] = detection
                new_packages += 1
                logger.info(f"📦 New unique package detected: {package_id[:8]}")

        if new_packages > 0 or updated_packages > 0:
            self._last_package_detection_time = current_time
            logger.info(
                f"📦 Package detection summary - New: {new_packages}, Updated: {updated_packages}, "
                f"Total unique packages: {len(self._detected_packages)}"
            )

        return new_packages

    def _create_overlay(
        self, frame_bgr: np.ndarray, face_count: int, package_count: int
    ) -> np.ndarray:
        """
        Create video overlay with face count, package count, and thumbnail grid.

        Args:
            frame_bgr: Original frame in BGR format
            face_count: Number of faces in time window
            package_count: Number of packages in time window

        Returns:
            Frame with overlay applied
        """
        height, width = frame_bgr.shape[:2]
        overlay_width = 200  # Width of right panel
        grid_cols = 2  # Number of columns in thumbnail grid

        # Create a copy to draw on
        frame_with_overlay = frame_bgr.copy()

        # Draw package bounding boxes on the frame
        for package in self._detected_packages.values():
            x, y, w, h = package.bbox
            # Ensure coordinates are integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            # Draw blue rectangle for packages
            cv2.rectangle(frame_with_overlay, (x, y), (x2, y2), (255, 0, 0), 2)
            # Draw package label
            label_text = f"Package {package.confidence:.2f}"
            cv2.putText(
                frame_with_overlay,
                label_text,
                (x, max(10, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw semi-transparent overlay panel on right side
        overlay = frame_with_overlay.copy()
        cv2.rectangle(
            overlay,
            (width - overlay_width, 0),
            (width, height),
            (40, 40, 40),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame_with_overlay, 0.3, 0, frame_with_overlay)

        # Draw header text
        header_text = "SECURITY CAMERA"
        cv2.putText(
            frame_with_overlay,
            header_text,
            (width - overlay_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw face count
        count_text = f"Visitors (30m): {face_count}"
        cv2.putText(
            frame_with_overlay,
            count_text,
            (width - overlay_width + 10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw package count
        package_text = f"Packages (30m): {package_count}"
        cv2.putText(
            frame_with_overlay,
            package_text,
            (width - overlay_width + 10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw thumbnail grid
        if self._detected_faces:
            grid_start_y = 80
            grid_padding = 10
            thumb_size = self.thumbnail_size

            # Get most recent 12 faces sorted by last_seen
            recent_faces = sorted(
                self._detected_faces.values(), key=lambda f: f.last_seen, reverse=True
            )[:12]

            for idx, detection in enumerate(recent_faces):
                row = idx // grid_cols
                col = idx % grid_cols

                x_pos = width - overlay_width + 10 + col * (thumb_size + grid_padding)
                y_pos = grid_start_y + row * (thumb_size + grid_padding)

                # Check if we're still within the frame bounds
                if y_pos + thumb_size > height:
                    break

                # Draw thumbnail
                try:
                    frame_with_overlay[
                        y_pos : y_pos + thumb_size, x_pos : x_pos + thumb_size
                    ] = detection.face_image

                    # Draw detection count badge
                    if detection.detection_count > 1:
                        badge_text = f"{detection.detection_count}x"
                        badge_size = cv2.getTextSize(
                            badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
                        )[0]
                        badge_x = x_pos + thumb_size - badge_size[0] - 2
                        badge_y = y_pos + thumb_size - 2

                        # Draw badge background
                        cv2.rectangle(
                            frame_with_overlay,
                            (badge_x - 2, badge_y - badge_size[1] - 2),
                            (x_pos + thumb_size, y_pos + thumb_size),
                            (0, 0, 0),
                            -1,
                        )

                        # Draw badge text
                        cv2.putText(
                            frame_with_overlay,
                            badge_text,
                            (badge_x, badge_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                except Exception as e:
                    logger.debug(f"Failed to draw thumbnail: {e}")
                    continue

        timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame_with_overlay,
            timestamp_text,
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame_with_overlay

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        try:
            current_time = time.time()

            # Convert frame to BGR (OpenCV format)
            frame_rgb = frame.to_ndarray(format="rgb24")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Clean up old faces and packages
            self._cleanup_old_faces(current_time)
            self._cleanup_old_packages(current_time)

            # Detect and store new faces
            await self._detect_and_store_faces(frame_bgr, current_time)

            # Detect and store new packages
            await self._detect_and_store_packages(frame_bgr, current_time)

            # Create overlay with stats and thumbnails
            frame_with_overlay = self._create_overlay(
                frame_bgr, len(self._detected_faces), len(self._detected_packages)
            )

            # Convert back to RGB and then to av.VideoFrame
            frame_rgb_overlay = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
            processed_frame = av.VideoFrame.from_ndarray(
                frame_rgb_overlay, format="rgb24"
            )

            # Publish the processed frame
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)

    async def process_video(
        self,
        incoming_track: aiortc.mediastreams.MediaStreamTrack,
        participant: Any,
        shared_forwarder=None,
    ):
        """
        Set up video processing pipeline.

        Args:
            incoming_track: The incoming video track to process
            participant: Participant information
            shared_forwarder: Optional shared VideoForwarder
        """
        logger.info("✅ Security Camera process_video starting")

        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Security Camera subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            self._video_forwarder.add_frame_handler(
                self._process_and_add_frame, fps=float(self.fps), name="security_camera"
            )
        else:
            self._video_forwarder = VideoForwarder(
                incoming_track,
                max_buffer=30,
                fps=self.fps,
                name="security_camera_forwarder",
            )
            self._video_forwarder.add_frame_handler(self._process_and_add_frame)

        logger.info("✅ Security Camera video processing pipeline started")

    def publish_video_track(self):
        """Return the video track for publishing."""
        logger.info("📹 publish_video_track called")
        return self._video_track

    def state(self) -> Dict[str, Any]:
        """
        Return current state for LLM context.

        Returns:
            Dictionary with visitor count, package count, and timing info
        """
        current_time = time.time()
        self._cleanup_old_faces(current_time)
        self._cleanup_old_packages(current_time)

        total_face_detections = sum(
            face.detection_count for face in self._detected_faces.values()
        )
        total_package_detections = sum(
            package.detection_count for package in self._detected_packages.values()
        )

        return {
            "unique_visitors": len(self._detected_faces),
            "total_face_detections": total_face_detections,
            "unique_packages": len(self._detected_packages),
            "total_package_detections": total_package_detections,
            "time_window_minutes": self.time_window // 60,
            "last_face_detection_time": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self._last_detection_time)
                )
                if self._last_detection_time > 0
                else "No detections yet"
            ),
            "last_package_detection_time": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(self._last_package_detection_time),
                )
                if self._last_package_detection_time > 0
                else "No detections yet"
            ),
        }

    def get_visitor_count(self) -> int:
        """
        Get the current unique visitor count (for function calling).

        Returns:
            Number of unique faces detected in the time window
        """
        current_time = time.time()
        self._cleanup_old_faces(current_time)
        return len(self._detected_faces)

    def get_visitor_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all visitors.

        Returns:
            List of visitor details
        """
        current_time = time.time()
        self._cleanup_old_faces(current_time)

        visitors = []
        for face in sorted(
            self._detected_faces.values(), key=lambda f: f.last_seen, reverse=True
        ):
            visitors.append(
                {
                    "face_id": face.face_id[:8],  # Shortened ID
                    "first_seen": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(face.first_seen)
                    ),
                    "last_seen": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(face.last_seen)
                    ),
                    "detection_count": face.detection_count,
                }
            )

        return visitors

    def get_package_count(self) -> int:
        """
        Get the current unique package count (for function calling).

        Returns:
            Number of unique packages detected in the time window
        """
        current_time = time.time()
        self._cleanup_old_packages(current_time)
        return len(self._detected_packages)

    def get_package_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all packages.

        Returns:
            List of package details
        """
        current_time = time.time()
        self._cleanup_old_packages(current_time)

        packages = []
        for package in sorted(
            self._detected_packages.values(),
            key=lambda p: p.last_seen,
            reverse=True,
        ):
            packages.append(
                {
                    "package_id": package.package_id[:8],
                    "first_seen": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(package.first_seen)
                    ),
                    "last_seen": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(package.last_seen)
                    ),
                    "detection_count": package.detection_count,
                    "confidence": package.confidence,
                }
            )

        return packages

    def close(self):
        """Clean up resources."""
        logger.info("🛑 Security Camera Processor closing")
        self.executor.shutdown(wait=False)
        self._detected_faces.clear()
        self._detected_packages.clear()
        logger.info("🛑 Security Camera Processor closed")
