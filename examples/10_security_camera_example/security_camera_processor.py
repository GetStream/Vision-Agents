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
from PIL import Image

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


class SecurityCameraProcessor(
    AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin
):
    """
    Security camera processor that detects and recognizes faces in a grid overlay.

    This processor:
    - Detects faces in real-time using OpenCV
    - Uses face_recognition library to identify unique individuals
    - Prevents duplicate entries for the same person
    - Maintains a 30-minute sliding window of unique visitors
    - Displays visitor count and thumbnails in a grid overlay on the video

    Args:
        fps: Frame processing rate (default: 5)
        max_workers: Number of worker threads (default: 4)
        time_window: Time window in seconds to track faces (default: 1800 = 30 minutes)
        thumbnail_size: Size of face thumbnails in overlay (default: 80)
        detection_interval: Minimum seconds between face detections (default: 2)
        face_match_tolerance: Face recognition tolerance (default: 0.6, lower = stricter)
    """

    name = "security_camera"

    def __init__(
        self,
        fps: int = 5,
        max_workers: int = 4,
        time_window: int = 1800,
        thumbnail_size: int = 80,
        detection_interval: float = 2.0,
        face_match_tolerance: float = 0.6,
    ):
        super().__init__(interval=0, receive_audio=False, receive_video=True)

        self.fps = fps
        self.max_workers = max_workers
        self.time_window = time_window
        self.thumbnail_size = thumbnail_size
        self.detection_interval = detection_interval
        self.face_match_tolerance = face_match_tolerance

        # Storage for unique detected faces (keyed by face_id)
        self._detected_faces: Dict[str, FaceDetection] = {}
        self._last_detection_time = 0.0

        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="security_camera"
        )

        # Video track for publishing
        self._video_track: QueuedVideoTrack = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # Load OpenCV face detector
        self._face_cascade = None
        self._load_face_detector()

        logger.info("🎥 Security Camera Processor initialized")
        logger.info(f"📊 Time window: {time_window}s ({time_window // 60} minutes)")
        logger.info(f"🖼️ Thumbnail size: {thumbnail_size}x{thumbnail_size}")

    def _load_face_detector(self):
        """Load OpenCV Haar Cascade face detector."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✅ Face detector loaded")
        except Exception as e:
            logger.exception(f"❌ Failed to load face detector: {e}")
            raise

    def _cleanup_old_faces(self, current_time: float):
        """Remove faces that haven't been seen within the time window."""
        cutoff_time = current_time - self.time_window
        original_count = len(self._detected_faces)
        
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

    def _detect_faces_sync(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Synchronously detect faces and generate encodings using face_recognition.

        Args:
            frame_rgb: Frame in RGB format

        Returns:
            List of dictionaries with 'bbox' and 'encoding' keys
        """
        # Detect face locations (top, right, bottom, left)
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
            
            results.append({
                "bbox": (x, y, w, h),
                "encoding": encoding
            })
        
        return results

    def _find_matching_face(self, face_encoding: np.ndarray) -> Optional[str]:
        """
        Find if a face encoding matches any existing detected face.

        Args:
            face_encoding: Face encoding to match

        Returns:
            face_id of matching face, or None if no match
        """
        if not self._detected_faces:
            return None
        
        # Get all existing face encodings
        known_face_ids = list(self._detected_faces.keys())
        known_encodings = [
            self._detected_faces[face_id].face_encoding 
            for face_id in known_face_ids
        ]
        
        # Compare against all known faces
        matches = face_recognition.compare_faces(
            known_encodings, 
            face_encoding, 
            tolerance=self.face_match_tolerance
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

    def _create_overlay(
        self, frame_bgr: np.ndarray, face_count: int
    ) -> np.ndarray:
        """
        Create video overlay with face count and thumbnail grid.

        Args:
            frame_bgr: Original frame in BGR format
            face_count: Number of faces in time window

        Returns:
            Frame with overlay applied
        """
        height, width = frame_bgr.shape[:2]
        overlay_width = 200  # Width of right panel
        grid_cols = 2  # Number of columns in thumbnail grid

        # Create a copy to draw on
        frame_with_overlay = frame_bgr.copy()

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

        # Draw thumbnail grid
        if self._detected_faces:
            grid_start_y = 80
            grid_padding = 10
            thumb_size = self.thumbnail_size

            # Get most recent 12 faces sorted by last_seen
            recent_faces = sorted(
                self._detected_faces.values(),
                key=lambda f: f.last_seen,
                reverse=True
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

        # Draw timestamp
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
        """Process a video frame: detect faces and add overlay."""
        try:
            current_time = time.time()

            # Convert frame to BGR (OpenCV format)
            frame_rgb = frame.to_ndarray(format="rgb24")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Clean up old faces
            self._cleanup_old_faces(current_time)

            # Detect and store new faces
            await self._detect_and_store_faces(frame_bgr, current_time)

            # Create overlay with stats and thumbnails
            frame_with_overlay = self._create_overlay(
                frame_bgr, len(self._detected_faces)
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
            Dictionary with visitor count and timing info
        """
        current_time = time.time()
        self._cleanup_old_faces(current_time)

        total_detections = sum(
            face.detection_count for face in self._detected_faces.values()
        )

        return {
            "unique_visitors": len(self._detected_faces),
            "total_detections": total_detections,
            "time_window_minutes": self.time_window // 60,
            "last_detection_time": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self._last_detection_time)
                )
                if self._last_detection_time > 0
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
            self._detected_faces.values(), 
            key=lambda f: f.last_seen, 
            reverse=True
        ):
            visitors.append({
                "face_id": face.face_id[:8],  # Shortened ID
                "first_seen": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(face.first_seen)
                ),
                "last_seen": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(face.last_seen)
                ),
                "detection_count": face.detection_count,
            })
        
        return visitors

    def close(self):
        """Clean up resources."""
        logger.info("🛑 Security Camera Processor closing")
        self.executor.shutdown(wait=False)
        self._detected_faces.clear()
        logger.info("🛑 Security Camera Processor closed")

