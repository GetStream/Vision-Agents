import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from PIL import Image
import numpy as np
import cv2

from vision_agents.plugins import ultralytics
from squat_counter_processor import SquatCounterProcessor
from squat_events import SquatCompletedEvent
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent

logger = logging.getLogger(__name__)


class SquatYOLOProcessor(ultralytics.YOLOPoseProcessor):
    """
    Extended YOLO Pose Processor with integrated squat counting.
    
    Combines YOLO pose detection with biomechanical analysis for accurate
    squat counting and form feedback.
    """
    
    name = "squat_yolo"
    
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        imgsz: int = 512,
        device: str = "cpu",
        max_workers: int = 24,
        fps: int = 30,
        interval: int = 0,
        enable_hand_tracking: bool = False,
        enable_wrist_highlights: bool = False,
        min_squat_angle: float = 100.0,
        max_standing_angle: float = 160.0,
        avatar_path: Optional[str] = None,
        avatar_scale: float = 2.0,
        hide_skeleton: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
            max_workers=max_workers,
            fps=fps,
            interval=interval,
            enable_hand_tracking=enable_hand_tracking,
            enable_wrist_highlights=enable_wrist_highlights,
            *args,
            **kwargs,
        )
        
        # Store agent reference (set via _attach_agent)
        self._agent: Optional["Agent"] = None
        
        # Initialize squat counter (no callback needed - we'll use events)
        self.squat_counter = SquatCounterProcessor(
            min_squat_angle=min_squat_angle,
            max_standing_angle=max_standing_angle,
            confidence_threshold=conf_threshold,
        )
        
        # Load avatar image if provided
        self.avatar_image = None
        self.avatar_scale = avatar_scale
        self.hide_skeleton = hide_skeleton
        if avatar_path:
            self.avatar_image = self._load_avatar(avatar_path)
            if self.avatar_image is not None:
                logger.info(f"🎭 Avatar loaded from {avatar_path}")
        
        logger.info("💪 Squat YOLO Processor initialized with squat counting")
    
    def _load_avatar(self, avatar_path: str) -> Optional[np.ndarray]:
        """Load avatar image with transparency support."""
        try:
            # Load image with alpha channel
            avatar = cv2.imread(avatar_path, cv2.IMREAD_UNCHANGED)
            if avatar is None:
                logger.error(f"Failed to load avatar from {avatar_path}")
                return None
            
            # Ensure it has an alpha channel
            if avatar.shape[2] == 3:
                # Add alpha channel if missing
                avatar = cv2.cvtColor(avatar, cv2.COLOR_BGR2BGRA)
                avatar[:, :, 3] = 255  # Fully opaque
            
            return avatar
        except Exception as e:
            logger.error(f"Error loading avatar: {e}")
            return None
    
    def _overlay_avatar(self, frame: np.ndarray, pose_data: Dict[str, Any]) -> np.ndarray:
        """Overlay avatar on detected face."""
        if self.avatar_image is None or not pose_data.get("persons"):
            return frame
        
        try:
            # Get first person's keypoints
            person = pose_data["persons"][0]
            keypoints = person.get("keypoints", [])
            
            if len(keypoints) < 5:  # Need at least nose and eyes
                return frame
            
            # YOLO keypoint indices (COCO format)
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            
            # Check if face keypoints are detected with sufficient confidence
            if nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5:
                return frame
            
            # Calculate face center and size
            face_center_x = int((left_eye[0] + right_eye[0]) / 2)
            face_center_y = int((left_eye[1] + right_eye[1]) / 2)
            
            # Calculate face width based on eye distance
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            avatar_width = int(eye_distance * self.avatar_scale)
            
            if avatar_width <= 0:
                return frame
            
            # Resize avatar maintaining aspect ratio
            avatar_height = int(avatar_width * self.avatar_image.shape[0] / self.avatar_image.shape[1])
            resized_avatar = cv2.resize(self.avatar_image, (avatar_width, avatar_height))
            
            # Calculate position (center the avatar on the face)
            x1 = face_center_x - avatar_width // 2
            y1 = face_center_y - avatar_height // 2
            x2 = x1 + avatar_width
            y2 = y1 + avatar_height
            
            # Ensure avatar fits within frame bounds
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                # Clip to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Adjust avatar size to match clipped region
                avatar_width = x2 - x1
                avatar_height = y2 - y1
                if avatar_width <= 0 or avatar_height <= 0:
                    return frame
                resized_avatar = cv2.resize(self.avatar_image, (avatar_width, avatar_height))
            
            # Extract alpha channel for transparency
            if resized_avatar.shape[2] == 4:
                avatar_rgb = resized_avatar[:, :, :3]
                avatar_alpha = resized_avatar[:, :, 3] / 255.0
                
                # Get the region of interest from the frame
                roi = frame[y1:y2, x1:x2]
                
                # Blend avatar with frame using alpha channel
                for c in range(3):
                    roi[:, :, c] = (avatar_alpha * avatar_rgb[:, :, c] + 
                                   (1 - avatar_alpha) * roi[:, :, c])
                
                frame[y1:y2, x1:x2] = roi
            else:
                # No alpha channel, just paste it
                frame[y1:y2, x1:x2] = resized_avatar
            
            return frame
            
        except Exception as e:
            logger.error(f"Error overlaying avatar: {e}")
            return frame
    
    def _attach_agent(self, agent: "Agent") -> None:
        """Attach agent reference to access event manager."""
        self._agent = agent
        # Register the squat event with the agent's event manager
        if agent.events:
            agent.events.register(SquatCompletedEvent)
    
    def _process_pose_sync(
        self, frame_array: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Process pose and add squat counting logic.
        Overrides parent method to add squat analysis.
        """
        import time
        start_time = time.time()
        
        # Call parent's pose processing
        annotated_frame, pose_data = super()._process_pose_sync(frame_array)
        
        # If hiding skeleton, use original frame instead of annotated
        if self.hide_skeleton:
            annotated_frame = frame_array.copy()
        
        # Overlay avatar on face if enabled
        if self.avatar_image is not None:
            annotated_frame = self._overlay_avatar(annotated_frame, pose_data)
        
        # Process squat counting
        squat_data = self.squat_counter.process_pose_data(pose_data)
        
        # Emit event if squat was completed (message indicates completion)
        if squat_data.get("message") and self._agent and self._agent.events:
            event = SquatCompletedEvent(
                plugin_name="squat_counter",
                rep_count=squat_data["rep_count"],
                knee_angle=squat_data.get("knee_angle", 0.0),
                timestamp=squat_data.get("timestamp", 0.0),
            )
            self._agent.events.send(event)
            logger.debug(f"📤 Emitted SquatCompletedEvent: Rep #{squat_data['rep_count']}")
        
        # Add squat info to frame
        annotated_frame = self._draw_squat_info(annotated_frame, squat_data)
        
        # Merge squat data into pose data
        pose_data["squat"] = squat_data
        
        # Log processing time and rep count
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        if squat_data.get("message"):
            logger.info(f"⏱️  Processing latency: {processing_time:.1f}ms | Rep #{squat_data['rep_count']} detected")
        
        return annotated_frame, pose_data
    
    def _draw_squat_info(
        self, 
        frame: np.ndarray, 
        squat_data: Dict[str, Any]
    ) -> np.ndarray:
        """Draw squat counter and info on frame."""
        height, width = frame.shape[:2]
        
        # Draw rep counter (top left)
        rep_text = f"Reps: {squat_data['rep_count']}"
        cv2.putText(
            frame,
            rep_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,  # Thicker line for bold effect
        )
        
        # Draw phase (top left, below counter)
        phase_text = f"Phase: {squat_data['phase']}"
        phase_color = self._get_phase_color(squat_data['phase'])
        cv2.putText(
            frame,
            phase_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            phase_color,
            2,
        )
        
        # Draw knee angle if available
        if squat_data.get('knee_angle'):
            angle_text = f"Knee: {squat_data['knee_angle']:.0f}°"
            cv2.putText(
                frame,
                angle_text,
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        
        # Draw form issues (bottom left)
        if squat_data.get('form_issues'):
            y_offset = height - 100
            for issue in squat_data['form_issues'][:3]:  # Max 3 issues
                cv2.putText(
                    frame,
                    f"⚠ {issue}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
                y_offset += 30
        
        # Draw message if available (center top)
        if squat_data.get('message'):
            message = squat_data['message']
            # Calculate text size for centering
            (text_width, text_height), _ = cv2.getTextSize(
                message,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                4
            )
            x = (width - text_width) // 2
            
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (x - 10, 10),
                (x + text_width + 10, 50 + text_height),
                (0, 255, 0),
                -1,
            )
            
            # Draw text
            cv2.putText(
                frame,
                message,
                (x, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                4,  # Thicker line for bold effect
            )
        
        return frame
    
    def _get_phase_color(self, phase: str) -> tuple:
        """Get color for phase visualization."""
        colors = {
            "standing": (0, 255, 0),      # Green
            "descending": (0, 255, 255),  # Yellow
            "bottom": (0, 165, 255),      # Orange
            "ascending": (255, 255, 0),   # Cyan
        }
        return colors.get(phase, (255, 255, 255))
    
    def state(self) -> Dict[str, Any]:
        """Return current state including squat counter state."""
        return {
            "processor": "squat_yolo",
            "squat_counter": self.squat_counter.state(),
        }
    
    def reset_counter(self):
        """Reset the squat counter."""
        self.squat_counter.reset()

