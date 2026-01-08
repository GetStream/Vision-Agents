import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiortc
import av
import cv2
import face_recognition
import numpy as np

from pathlib import Path

from vision_agents.core.events.base import PluginBaseEvent
from vision_agents.core.events.manager import EventManager
from vision_agents.core.processors.base_processor import (
    VideoProcessorPublisher,
)
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)

# Constants
OVERLAY_WIDTH = 200
GRID_COLS = 2
MAX_THUMBNAILS = 12
PICKUP_THRESHOLD_SECONDS = 15.0
PICKUP_MAX_AGE_SECONDS = 300.0
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class PersonDetectedEvent(PluginBaseEvent):
    """Event emitted when a person/face is detected."""

    type: str = field(default="security.person_detected", init=False)
    face_id: str = ""
    is_new: bool = False
    detection_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class PackageDetectedEvent(PluginBaseEvent):
    """Event emitted when a package is detected."""

    type: str = field(default="security.package_detected", init=False)
    package_id: str = ""
    is_new: bool = False
    detection_count: int = 1
    confidence: float = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class PackageDisappearedEvent(PluginBaseEvent):
    """Event emitted when a package disappears from the frame."""

    type: str = field(default="security.package_disappeared", init=False)
    package_id: str = ""
    confidence: float = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class PersonDisappearedEvent(PluginBaseEvent):
    """Event emitted when a person disappears from the frame."""

    type: str = field(default="security.person_disappeared", init=False)
    face_id: str = ""
    name: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


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
    name: Optional[str] = None  # Name if this is a known face
    disappeared_at: Optional[float] = None  # When this face left the frame


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
    disappeared_at: Optional[float] = None  # When this package left the frame


@dataclass
class KnownFace:
    """Represents a known/registered face."""

    name: str
    face_encoding: np.ndarray
    registered_at: float


@dataclass
class ActivityLogEntry:
    """Represents an entry in the activity log."""

    timestamp: float
    event_type: str  # "person_detected", "package_detected", "person_left", etc.
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityCameraProcessor(VideoProcessorPublisher, Warmable[Optional[Any]]):
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
        package_min_area_ratio: float = 0.01,  # Minimum area as ratio of frame (1% of frame)
        package_max_area_ratio: float = 0.9,  # Maximum area as ratio of frame (50% of frame)
    ):
        self.fps = fps
        self.max_workers = max_workers
        self.time_window = time_window
        self.thumbnail_size = thumbnail_size
        self.detection_interval = detection_interval
        self.face_match_tolerance = face_match_tolerance
        self.package_detection_interval = package_detection_interval
        self.package_fps = package_fps
        self.package_conf_threshold = package_conf_threshold
        self.package_min_area_ratio = package_min_area_ratio
        self.package_max_area_ratio = package_max_area_ratio

        # Storage for unique detected faces (keyed by face_id)
        self._detected_faces: Dict[str, FaceDetection] = {}
        self._last_detection_time = 0.0

        # Storage for unique detected packages (keyed by package_id)
        self._detected_packages: Dict[str, PackageDetection] = {}
        self._last_package_detection_time = 0.0

        # Known faces database for named recognition
        self._known_faces: Dict[str, KnownFace] = {}

        # Activity log for event history
        self._activity_log: List[ActivityLogEntry] = []
        self._max_activity_log_entries = 100  # Keep last 100 events

        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="security_camera"
        )

        # Shutdown flag to prevent new tasks
        self._shutdown = False

        # Video track for publishing
        self._video_track: QueuedVideoTrack = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # Initialize YOLO model for package detection
        self.model_path = model_path
        self.device = device
        self.yolo_model: Optional[Any] = None
        # Package-related classes detected by the weights.pt model
        self.package_detect_classes = [
            "package",
            "parcel",
            "box",
            "boxes",
            "Box",
            "Boxes",
            "Box_broken",
            "Cardboard",
            "Cardboards",
            "cardboard",
            "Open_package",
            "damaged box",
            "good-parcel",
            "Parcel",
            "Package",
            "suitcase",
            "backpack",
            "handbag",
        ]

        # Event manager for detection events
        self.events = EventManager()
        self.events.register(PersonDetectedEvent)
        self.events.register(PackageDetectedEvent)
        self.events.register(PackageDisappearedEvent)
        self.events.register(PersonDisappearedEvent)

        logger.info("🎥 Security Camera Processor initialized")
        logger.info(f"📊 Time window: {time_window}s ({time_window // 60} minutes)")
        logger.info(f"🖼️ Thumbnail size: {thumbnail_size}x{thumbnail_size}")
        logger.info(
            f"📦 Package detection: {package_fps} FPS, interval: {package_detection_interval}s"
        )

    def _format_timestamp(self, timestamp: float) -> str:
        """Format a Unix timestamp as a human-readable string."""
        return time.strftime(TIMESTAMP_FORMAT, time.localtime(timestamp))

    def _cleanup_old_items(
        self, items: Dict[str, Any], current_time: float, item_type: str
    ) -> int:
        """Remove items whose last_seen is older than the time window.
        
        Returns the number of items removed.
        """
        cutoff_time = current_time - self.time_window
        to_remove = [
            item_id for item_id, item in items.items() if item.last_seen < cutoff_time
        ]
        for item_id in to_remove:
            del items[item_id]
        if to_remove:
            logger.debug(f"🧹 Cleaned up {len(to_remove)} old {item_type}(s)")
        return len(to_remove)

    async def on_warmup(self) -> Optional[Any]:
        """Load YOLO model for package detection."""
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

            yolo_model = await loop.run_in_executor(self.executor, load_yolo_model)
            logger.info(f"✅ YOLO model loaded: {self.model_path} on {self.device}")
            return yolo_model
        except Exception as e:
            logger.warning(f"⚠️ Failed to load YOLO model: {e}")
            logger.warning("⚠️ Package detection will be disabled")
            return None

    def on_warmed_up(self, resource: Optional[Any]) -> None:
        """Set the loaded YOLO model to the instance."""
        self.yolo_model = resource

    def _cleanup_old_faces(self, current_time: float) -> int:
        """Remove faces older than the time window."""
        return self._cleanup_old_items(self._detected_faces, current_time, "face")

    def _cleanup_old_packages(self, current_time: float) -> int:
        """Remove packages older than the time window."""
        return self._cleanup_old_items(self._detected_packages, current_time, "package")

    def _check_for_picked_up_packages(self, current_time: float):
        """Check if any packages have disappeared (picked up).

        A package is considered "picked up" if it hasn't been seen for PICKUP_THRESHOLD_SECONDS
        but was detected within the last PICKUP_MAX_AGE_SECONDS.
        """
        packages_picked_up = []

        for package_id, package in list(self._detected_packages.items()):
            time_since_seen = current_time - package.last_seen
            package_age = current_time - package.first_seen

            # Package disappeared recently (not seen for threshold, but was active recently)
            if (
                PICKUP_THRESHOLD_SECONDS < time_since_seen < PICKUP_MAX_AGE_SECONDS
                and package_age < PICKUP_MAX_AGE_SECONDS
            ):
                packages_picked_up.append(package)

        for package in packages_picked_up:
            # Find who was present when the package disappeared
            picker = self._find_person_present_at(package.last_seen)
            picker_name = (
                picker.name
                if picker and picker.name
                else (picker.face_id[:8] if picker else "unknown person")
            )

            logger.info(
                f"📦 Package {package.package_id[:8]} was picked up by {picker_name}"
            )

            # Log activity
            self._log_activity(
                event_type="package_picked_up",
                description=f"Package picked up by {picker_name}",
                details={
                    "package_id": package.package_id[:8],
                    "picked_up_by": picker_name,
                    "picker_face_id": picker.face_id[:8] if picker else None,
                    "picker_is_known": picker.name is not None if picker else False,
                },
            )

            # Remove the package from tracking
            del self._detected_packages[package.package_id]

    def _find_person_present_at(self, timestamp: float) -> Optional[FaceDetection]:
        """Find who was present around a given timestamp.

        Returns the person who was most recently seen around that time.
        """
        window = 10.0  # Look within 10 seconds of the timestamp

        candidates = []
        for face in self._detected_faces.values():
            # Check if person was seen around that time
            if abs(face.last_seen - timestamp) < window:
                candidates.append(face)

        if not candidates:
            return None

        # Return the most recently seen person
        return max(candidates, key=lambda f: f.last_seen)

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

    def _find_known_face_name(self, face_encoding: np.ndarray) -> Optional[str]:
        """Check if face matches any known/registered face and return the name."""
        if not self._known_faces:
            return None

        known_names = list(self._known_faces.keys())
        known_encodings = [
            self._known_faces[name].face_encoding for name in known_names
        ]

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=self.face_match_tolerance
        )

        for i, is_match in enumerate(matches):
            if is_match:
                return known_names[i]

        return None

    def _log_activity(
        self,
        event_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an entry to the activity log."""
        entry = ActivityLogEntry(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            details=details or {},
        )
        self._activity_log.append(entry)

        # Trim log if too long
        if len(self._activity_log) > self._max_activity_log_entries:
            self._activity_log = self._activity_log[-self._max_activity_log_entries :]

    async def _detect_and_store_faces(
        self, frame_bgr: np.ndarray, current_time: float
    ) -> int:
        """
        Detect faces in frame and store new unique faces or update existing ones.

        Returns:
            Number of new unique faces detected
        """
        if self._shutdown:
            return 0

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
        faces_seen_this_frame: set[str] = set()

        for face_data in detected_faces:
            x, y, w, h = face_data["bbox"]
            face_encoding = face_data["encoding"]

            # Extract face thumbnail (convert back to BGR for storage)
            face_roi = frame_bgr[y : y + h, x : x + w]
            face_thumbnail = cv2.resize(
                face_roi, (self.thumbnail_size, self.thumbnail_size)
            )

            # Check if this is a known/registered face
            known_name = self._find_known_face_name(face_encoding)

            # Check if this face matches any existing face in current session
            matching_face_id = self._find_matching_face(face_encoding)

            if matching_face_id:
                # Update existing face
                face_detection = self._detected_faces[matching_face_id]
                faces_seen_this_frame.add(matching_face_id)
                face_detection.last_seen = current_time
                face_detection.bbox = (x, y, w, h)
                # Update thumbnail to latest image
                face_detection.face_image = face_thumbnail
                # Update name if we now recognize them
                if known_name and not face_detection.name:
                    face_detection.name = known_name
                
                # Only increment count and emit event if they returned after disappearing
                # This ensures each visit (disappear → return) is counted independently
                if face_detection.disappeared_at is not None:
                    # They left and came back - increment count and emit event
                    # After clearing disappeared_at, if they disappear and return again,
                    # the cycle repeats and count increments again
                    face_detection.detection_count += 1
                    updated_faces += 1
                    display_name = face_detection.name or matching_face_id[:8]
                    logger.info(
                        f"👤 Returning visitor: {display_name} (visit #{face_detection.detection_count})"
                    )
                    self.events.send(
                        PersonDetectedEvent(
                            plugin_name="security_camera",
                            face_id=display_name,
                            is_new=False,
                            detection_count=face_detection.detection_count,
                            first_seen=self._format_timestamp(face_detection.first_seen),
                            last_seen=self._format_timestamp(current_time),
                        )
                    )
                    face_detection.disappeared_at = None  # Clear disappeared flag so next disappearance can be tracked
                else:
                    # Continuously present - just update silently, don't increment count
                    display_name = face_detection.name or matching_face_id[:8]
                    logger.debug(
                        f"🔄 Updated existing face {display_name} "
                        f"(continuously present, entry count: {face_detection.detection_count})"
                    )
                # If disappeared_at is None, they're continuously present - no event
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
                    name=known_name,  # Will be None if not recognized
                    disappeared_at=None,
                )
                self._detected_faces[face_id] = detection
                faces_seen_this_frame.add(face_id)
                new_faces += 1

                display_name = known_name or face_id[:8]
                logger.info(f"👤 New unique visitor detected: {display_name}")

                # Log activity
                self._log_activity(
                    event_type="person_arrived",
                    description=f"New person arrived: {display_name}",
                    details={
                        "face_id": face_id[:8],
                        "name": known_name,
                        "is_known": known_name is not None,
                    },
                )

                # Emit event for new visitor
                self.events.send(
                    PersonDetectedEvent(
                        plugin_name="security_camera",
                        face_id=display_name,
                        is_new=True,
                        detection_count=1,
                        first_seen=self._format_timestamp(current_time),
                        last_seen=self._format_timestamp(current_time),
                    )
                )

        # Mark faces that weren't seen this frame as disappeared
        # If disappeared_at is None, they were continuously present and just disappeared
        # If disappeared_at is already set, they're still disappeared (no change needed)
        for face_id, face_detection in self._detected_faces.items():
            if face_id not in faces_seen_this_frame:
                if face_detection.disappeared_at is None:
                    # They were present, now disappeared - mark the disappearance time
                    # This will trigger count increment when they return
                    face_detection.disappeared_at = current_time
                    display_name = face_detection.name or face_id[:8]
                    logger.info(f"👤 Person left: {display_name}")
                    
                    # Log activity
                    self._log_activity(
                        event_type="person_left",
                        description=f"Person left: {display_name}",
                        details={
                            "face_id": face_id[:8],
                            "name": face_detection.name,
                            "is_known": face_detection.name is not None,
                        },
                    )
                    
                    # Emit event
                    self.events.send(
                        PersonDisappearedEvent(
                            plugin_name="security_camera",
                            face_id=display_name,
                            name=face_detection.name,
                            first_seen=self._format_timestamp(face_detection.first_seen),
                            last_seen=self._format_timestamp(current_time),
                        )
                    )

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

            # Log all detections for debugging
            detected_classes = {}
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = class_names[cls_id]
                detected_classes[class_name] = detected_classes.get(class_name, 0) + 1

            if detected_classes:
                logger.info(
                    f"🔍 YOLO detected {len(boxes)} objects: {', '.join(f'{k}({v})' for k, v in detected_classes.items())}"
                )

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name_original = class_names[cls_id]
                class_name = class_name_original.lower()

                # Lowercase detect_class for case-insensitive matching
                matches_package_class = any(
                    detect_class.lower() in class_name
                    for detect_class in self.package_detect_classes
                )

                if matches_package_class:
                    x_min, y_min, x_max, y_max = box

                    x_min = int(max(0, min(x_min, width - 1)))
                    y_min = int(max(0, min(y_min, height - 1)))
                    x_max = int(max(x_min + 1, min(x_max, width)))
                    y_max = int(max(y_min + 1, min(y_max, height)))

                    x = x_min
                    y = y_min
                    w = x_max - x_min
                    h = y_max - y_min

                    # Filter by size to exclude walls and very small detections
                    frame_area = width * height
                    detection_area = w * h
                    area_ratio = detection_area / frame_area

                    if area_ratio < self.package_min_area_ratio:
                        logger.debug(
                            f"⏭️ Skipped small detection: {class_name_original} "
                            f"(area: {area_ratio:.2%} < {self.package_min_area_ratio:.2%})"
                        )
                        continue

                    if area_ratio > self.package_max_area_ratio:
                        logger.debug(
                            f"⏭️ Skipped large detection (likely wall/background): {class_name_original} "
                            f"(area: {area_ratio:.2%} > {self.package_max_area_ratio:.2%})"
                        )
                        continue

                    all_detections.append(
                        {
                            "bbox": (x, y, w, h),
                            "confidence": float(conf),
                            "label": class_name_original,
                        }
                    )
                    logger.info(
                        f"📦 Matched package class: {class_name_original} (confidence: {conf:.2f}, area: {area_ratio:.2%})"
                    )
                else:
                    logger.debug(
                        f"⏭️ Skipped non-package class: {class_name_original} (confidence: {conf:.2f})"
                    )

        except Exception as e:
            logger.warning(f"⚠️ Failed to detect packages with YOLO: {e}")

        if all_detections:
            logger.info(f"📦 Found {len(all_detections)} package-like objects")
        elif detected_classes:
            logger.debug(
                f"📦 No packages matched from {len(boxes)} detections. "
                f"Looking for: {', '.join(self.package_detect_classes[:5])}..."
            )

        return all_detections

    async def _detect_and_store_packages(
        self, frame_bgr: np.ndarray, current_time: float
    ) -> int:
        """
        Detect packages in frame and store new unique packages or update existing ones.

        Returns:
            Number of new unique packages detected
        """
        if self._shutdown:
            return 0

        if not self.yolo_model:
            return 0

        # Check if enough time has passed since last detection
        if (
            current_time - self._last_package_detection_time
            < self.package_detection_interval
        ):
            return 0

        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        detected_packages = await loop.run_in_executor(
            self.executor, self._detect_packages_sync, frame_rgb
        )

        new_packages = 0
        updated_packages = 0
        packages_seen_this_frame: set[str] = set()

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
                packages_seen_this_frame.add(matching_package_id)
                package_detection.last_seen = current_time
                package_detection.bbox = (x, y, w, h)
                package_detection.confidence = max(
                    package_detection.confidence, confidence
                )
                package_detection.package_image = package_thumbnail
                
                # Only increment count and emit event if package returned after disappearing
                if package_detection.disappeared_at is not None:
                    # Package left and came back - increment count and emit arrival event
                    package_detection.detection_count += 1
                    updated_packages += 1
                    logger.info(
                        f"📦 Package returned: {matching_package_id[:8]} (visit #{package_detection.detection_count})"
                    )
                    self.events.send(
                        PackageDetectedEvent(
                            plugin_name="security_camera",
                            package_id=matching_package_id[:8],
                            is_new=False,
                            detection_count=package_detection.detection_count,
                            confidence=package_detection.confidence,
                            first_seen=self._format_timestamp(package_detection.first_seen),
                            last_seen=self._format_timestamp(current_time),
                        )
                    )
                    package_detection.disappeared_at = None  # Clear disappeared flag
                else:
                    # Continuously present - just update silently, don't increment count or emit event
                    logger.debug(
                        f"🔄 Updated existing package {matching_package_id[:8]} "
                        f"(continuously present, entry count: {package_detection.detection_count})"
                    )
                # If disappeared_at is None, package is continuously present - no event
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
                    disappeared_at=None,
                )
                self._detected_packages[package_id] = detection
                packages_seen_this_frame.add(package_id)
                new_packages += 1
                logger.info(f"📦 New unique package detected: {package_id[:8]}")

                # Log activity
                self._log_activity(
                    event_type="package_arrived",
                    description=f"New package detected (confidence: {confidence:.2f})",
                    details={
                        "package_id": package_id[:8],
                        "confidence": confidence,
                    },
                )

                # Emit event for new package
                self.events.send(
                    PackageDetectedEvent(
                        plugin_name="security_camera",
                        package_id=package_id[:8],
                        is_new=True,
                        detection_count=1,
                        confidence=confidence,
                        first_seen=self._format_timestamp(current_time),
                        last_seen=self._format_timestamp(current_time),
                    )
                )

        # Mark packages that weren't seen this frame as disappeared
        for package_id, package_detection in self._detected_packages.items():
            if package_id not in packages_seen_this_frame:
                if package_detection.disappeared_at is None:
                    # First time disappearing - mark it and emit event
                    package_detection.disappeared_at = current_time
                    logger.info(
                        f"📦 Package disappeared: {package_id[:8]} (confidence: {package_detection.confidence:.2f})"
                    )
                    self.events.send(
                        PackageDisappearedEvent(
                            plugin_name="security_camera",
                            package_id=package_id[:8],
                            confidence=package_detection.confidence,
                            first_seen=self._format_timestamp(package_detection.first_seen),
                            last_seen=self._format_timestamp(current_time),
                        )
                    )

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

        # Create a copy to draw on
        frame_with_overlay = frame_bgr.copy()

        # Draw face bounding boxes on the frame (only for currently visible faces)
        for face in self._detected_faces.values():
            # Only draw if face hasn't disappeared (disappeared_at is None)
            if face.disappeared_at is not None:
                continue
            x, y, w, h = face.bbox
            # Ensure coordinates are integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            # Draw green rectangle for faces
            cv2.rectangle(frame_with_overlay, (x, y), (x2, y2), (0, 255, 0), 2)
            # Draw face label
            display_name = face.name or face.face_id[:8]
            label_text = f"{display_name}"
            cv2.putText(
                frame_with_overlay,
                label_text,
                (x, max(10, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw package bounding boxes on the frame (only for currently visible packages)
        for package in self._detected_packages.values():
            # Only draw if package hasn't disappeared (disappeared_at is None)
            if package.disappeared_at is not None:
                continue
            x, y, w, h = package.bbox
            # Ensure coordinates are integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            # Draw brighter blue rectangle for packages (BGR: brighter blue)
            cv2.rectangle(frame_with_overlay, (x, y), (x2, y2), (255, 150, 150), 2)
            # Draw package label in brighter blue
            label_text = f"Package {package.confidence:.2f}"
            cv2.putText(
                frame_with_overlay,
                label_text,
                (x, max(10, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 150, 150),
                1,
                cv2.LINE_AA,
            )

        # Draw semi-transparent overlay panel on right side
        overlay = frame_with_overlay.copy()
        cv2.rectangle(
            overlay,
            (width - OVERLAY_WIDTH, 0),
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
            (width - OVERLAY_WIDTH + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Calculate currently visible counts
        visible_faces = sum(
            1 for f in self._detected_faces.values() if f.disappeared_at is None
        )
        visible_packages = sum(
            1 for p in self._detected_packages.values() if p.disappeared_at is None
        )

        # Draw face count with visible indicator
        count_text = f"Visitors: {visible_faces}/{face_count}"
        cv2.putText(
            frame_with_overlay,
            count_text,
            (width - OVERLAY_WIDTH + 10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw package count with visible indicator (brighter blue)
        package_text = f"Packages: {visible_packages}/{package_count}"
        cv2.putText(
            frame_with_overlay,
            package_text,
            (width - OVERLAY_WIDTH + 10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 150, 150),
            1,
            cv2.LINE_AA,
        )

        # Draw legend
        legend_y = 90
        # Green square for faces
        cv2.rectangle(
            frame_with_overlay,
            (width - OVERLAY_WIDTH + 10, legend_y - 8),
            (width - OVERLAY_WIDTH + 20, legend_y + 2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame_with_overlay,
            "Person",
            (width - OVERLAY_WIDTH + 25, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        # Blue square for packages
        cv2.rectangle(
            frame_with_overlay,
            (width - OVERLAY_WIDTH + 80, legend_y - 8),
            (width - OVERLAY_WIDTH + 90, legend_y + 2),
            (255, 150, 150),
            -1,
        )
        cv2.putText(
            frame_with_overlay,
            "Package",
            (width - OVERLAY_WIDTH + 95, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Draw thumbnail grid (faces and packages combined)
        grid_start_y = 105  # Start below the legend
        grid_padding = 10
        thumb_size = self.thumbnail_size

        # Combine faces and packages, sorted by last_seen
        all_detections = []
        
        # Add faces
        for face in self._detected_faces.values():
            all_detections.append({
                "type": "face",
                "image": face.face_image,
                "last_seen": face.last_seen,
                "detection_count": face.detection_count,
                "name": face.name or face.face_id[:8],
            })
        
        # Add packages
        for package in self._detected_packages.values():
            all_detections.append({
                "type": "package",
                "image": package.package_image,
                "last_seen": package.last_seen,
                "detection_count": package.detection_count,
                "package_id": package.package_id[:8],
                "confidence": package.confidence,
            })
        
        # Sort by last_seen (most recent first) and take top MAX_THUMBNAILS
        recent_detections = sorted(
            all_detections, key=lambda d: d["last_seen"], reverse=True
        )[:MAX_THUMBNAILS]

        for idx, detection in enumerate(recent_detections):
            row = idx // GRID_COLS
            col = idx % GRID_COLS

            x_pos = width - OVERLAY_WIDTH + 10 + col * (thumb_size + grid_padding)
            y_pos = grid_start_y + row * (thumb_size + grid_padding)

            # Check if we're still within the frame bounds
            if y_pos + thumb_size > height:
                break

            # Draw thumbnail
            try:
                frame_with_overlay[
                    y_pos : y_pos + thumb_size, x_pos : x_pos + thumb_size
                ] = detection["image"]

                # Draw colored border to distinguish type
                border_color = (0, 255, 0) if detection["type"] == "face" else (255, 150, 150)  # Green for faces, blue for packages
                cv2.rectangle(
                    frame_with_overlay,
                    (x_pos, y_pos),
                    (x_pos + thumb_size, y_pos + thumb_size),
                    border_color,
                    2,
                )

                # Draw detection count badge
                if detection["detection_count"] > 1:
                    badge_text = f"{detection['detection_count']}x"
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

        timestamp_text = self._format_timestamp(time.time())
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

            # Check if any packages were picked up
            self._check_for_picked_up_packages(current_time)

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
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Set up video processing pipeline.

        Args:
            track: The incoming video track to process
            participant_id: Participant ID
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
                track,
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
        
        # Count currently visible (not disappeared) items
        currently_visible_visitors = sum(
            1 for f in self._detected_faces.values() if f.disappeared_at is None
        )
        currently_visible_packages = sum(
            1 for p in self._detected_packages.values() if p.disappeared_at is None
        )

        return {
            "unique_visitors": len(self._detected_faces),
            "currently_visible_visitors": currently_visible_visitors,
            "total_face_detections": total_face_detections,
            "unique_packages": len(self._detected_packages),
            "currently_visible_packages": currently_visible_packages,
            "total_package_detections": total_package_detections,
            "time_window_minutes": self.time_window // 60,
            "last_face_detection_time": (
                self._format_timestamp(self._last_detection_time)
                if self._last_detection_time > 0
                else "No detections yet"
            ),
            "last_package_detection_time": (
                self._format_timestamp(self._last_package_detection_time)
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
                    "name": face.name,  # Will be None if unknown
                    "is_known": face.name is not None,
                    "first_seen": self._format_timestamp(face.first_seen),
                    "last_seen": self._format_timestamp(face.last_seen),
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
                    "first_seen": self._format_timestamp(package.first_seen),
                    "last_seen": self._format_timestamp(package.last_seen),
                    "detection_count": package.detection_count,
                    "confidence": package.confidence,
                }
            )

        return packages

    def get_activity_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent activity log entries.

        Args:
            limit: Maximum number of entries to return (default: 20)

        Returns:
            List of activity log entries, most recent first
        """
        entries = []
        for entry in reversed(self._activity_log[-limit:]):
            entries.append(
                {
                    "timestamp": self._format_timestamp(entry.timestamp),
                    "event_type": entry.event_type,
                    "description": entry.description,
                    "details": entry.details,
                }
            )
        return entries

    def register_known_face(self, name: str, face_encoding: np.ndarray) -> bool:
        """
        Register a face encoding with a name for future recognition.

        Args:
            name: Name to associate with the face
            face_encoding: 128-dimensional face encoding from face_recognition

        Returns:
            True if registered successfully
        """
        self._known_faces[name] = KnownFace(
            name=name,
            face_encoding=face_encoding,
            registered_at=time.time(),
        )
        logger.info(f"✅ Registered known face: {name}")

        # Log activity
        self._log_activity(
            event_type="face_registered",
            description=f"Registered new known face: {name}",
            details={"name": name},
        )

        return True

    def register_current_face_as(self, name: str) -> Dict[str, Any]:
        """
        Register the most recently detected face with a name.
        Useful for "remember me as [name]" functionality.

        Args:
            name: Name to associate with the face

        Returns:
            Dict with success status and message
        """
        if not self._detected_faces:
            return {
                "success": False,
                "message": "No faces currently detected. Please make sure your face is visible.",
            }

        # Get the most recently seen face
        most_recent_face = max(self._detected_faces.values(), key=lambda f: f.last_seen)

        # Register the face encoding
        self.register_known_face(name, most_recent_face.face_encoding)

        # Update the face detection with the name
        most_recent_face.name = name

        return {
            "success": True,
            "message": f"I'll remember you as {name}! Next time I see you, I'll recognize you.",
            "face_id": most_recent_face.face_id[:8],
        }

    def get_known_faces(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered known faces.

        Returns:
            List of known face info
        """
        return [
            {
                "name": face.name,
                "registered_at": self._format_timestamp(face.registered_at),
            }
            for face in self._known_faces.values()
        ]

    async def close(self):
        """Clean up resources."""
        logger.info("🛑 Security Camera Processor closing")
        self._shutdown = True
        
        # Stop video forwarder if it exists
        if self._video_forwarder is not None:
            try:
                await self._video_forwarder.stop()
            except Exception as e:
                logger.warning(f"Error stopping video forwarder: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        self._detected_faces.clear()
        self._detected_packages.clear()
        logger.info("🛑 Security Camera Processor closed")
