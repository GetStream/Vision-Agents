import asyncio
import logging
import math
from typing import Optional, Dict, Any, List, Callable, Union, Awaitable
from enum import Enum
import numpy as np
from PIL import Image
import cv2

from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    ImageProcessorMixin,
)
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

logger = logging.getLogger(__name__)


class SquatPhase(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class SquatCounterProcessor(AudioVideoProcessor, ImageProcessorMixin):
    """
    Intelligent squat counter that analyzes pose keypoints to detect and count squats.
    
    Uses biomechanical analysis of hip and knee angles to determine squat phases
    and count repetitions accurately.
    """
    
    name = "squat_counter"
    
    # YOLO pose keypoint indices (COCO format)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(
        self,
        interval: int = 0,
        min_squat_angle: float = 100.0,
        max_standing_angle: float = 160.0,
        confidence_threshold: float = 0.5,
        min_frames_in_phase: int = 3,
        on_squat_complete: Optional[Union[Callable[[int, float, float], None], Callable[[int, float, float], Awaitable[None]]]] = None,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=False)
        
        self.min_squat_angle = min_squat_angle
        self.max_standing_angle = max_standing_angle
        self.confidence_threshold = confidence_threshold
        self.min_frames_in_phase = min_frames_in_phase
        self.on_squat_complete = on_squat_complete
        
        # State tracking
        self.current_phase = SquatPhase.STANDING
        self.phase_frame_count = 0
        self.rep_count = 0
        self.last_knee_angle = None
        self.last_hip_angle = None
        
        # Form tracking
        self.form_issues: List[str] = []
        
        logger.info("💪 Squat Counter Processor initialized")
    
    def _calculate_angle(
        self, 
        point1: np.ndarray, 
        point2: np.ndarray, 
        point3: np.ndarray
    ) -> float:
        """
        Calculate angle between three points.
        point2 is the vertex of the angle.
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 180.0
        
        # Calculate angle in degrees
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _get_keypoint(
        self, 
        keypoints: List[List[float]], 
        index: int
    ) -> Optional[np.ndarray]:
        """Extract keypoint if confidence is high enough."""
        if index >= len(keypoints):
            return None
        
        x, y, conf = keypoints[index]
        if conf < self.confidence_threshold:
            return None
        
        return np.array([x, y])
    
    def _analyze_squat_position(
        self, 
        keypoints: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Analyze body position to determine squat phase and form.
        Returns dict with angles and phase information.
        """
        # Get key points for both sides
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        left_knee = self._get_keypoint(keypoints, self.LEFT_KNEE)
        left_ankle = self._get_keypoint(keypoints, self.LEFT_ANKLE)
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        right_knee = self._get_keypoint(keypoints, self.RIGHT_KNEE)
        right_ankle = self._get_keypoint(keypoints, self.RIGHT_ANKLE)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        # Calculate angles for both legs (use average)
        knee_angles = []
        hip_angles = []
        
        # Left leg knee angle (hip-knee-ankle)
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            knee_angles.append(left_knee_angle)
        
        # Right leg knee angle
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            knee_angles.append(right_knee_angle)
        
        # Left hip angle (shoulder-hip-knee)
        if left_shoulder is not None and left_hip is not None and left_knee is not None:
            left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
            hip_angles.append(left_hip_angle)
        
        # Right hip angle
        if right_shoulder is not None and right_hip is not None and right_knee is not None:
            right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
            hip_angles.append(right_hip_angle)
        
        if not knee_angles:
            return {"valid": False}
        
        avg_knee_angle = np.mean(knee_angles)
        avg_hip_angle = np.mean(hip_angles) if hip_angles else None
        
        return {
            "valid": True,
            "knee_angle": avg_knee_angle,
            "hip_angle": avg_hip_angle,
            "left_knee_angle": knee_angles[0] if len(knee_angles) > 0 else None,
            "right_knee_angle": knee_angles[1] if len(knee_angles) > 1 else None,
        }
    
    def _detect_form_issues(
        self,
        keypoints: List[List[float]],
        knee_angle: float
    ) -> List[str]:
        """Detect common squat form issues."""
        issues = []
        
        # Check if squat is deep enough
        if self.current_phase == SquatPhase.BOTTOM and knee_angle > 110:
            issues.append("Go deeper - aim for thighs parallel to ground")
        
        # Check knee alignment (knees shouldn't go too far forward)
        left_knee = self._get_keypoint(keypoints, self.LEFT_KNEE)
        left_ankle = self._get_keypoint(keypoints, self.LEFT_ANKLE)
        right_knee = self._get_keypoint(keypoints, self.RIGHT_KNEE)
        right_ankle = self._get_keypoint(keypoints, self.RIGHT_ANKLE)
        
        if left_knee is not None and left_ankle is not None:
            if left_knee[0] > left_ankle[0] + 50:  # Knee too far forward
                issues.append("Keep knees behind toes")
        
        if right_knee is not None and right_ankle is not None:
            if right_knee[0] < right_ankle[0] - 50:
                issues.append("Keep knees behind toes")
        
        return issues
    
    def _update_phase(
        self, 
        knee_angle: float
    ) -> Optional[str]:
        """
        Update squat phase based on knee angle.
        Returns message if rep completed or phase changed significantly.
        """
        import time
        message = None
        
        # Determine what phase we should be in based on angle
        if knee_angle >= self.max_standing_angle:
            target_phase = SquatPhase.STANDING
        elif knee_angle <= self.min_squat_angle:
            target_phase = SquatPhase.BOTTOM
        elif self.last_knee_angle is not None:
            # Determine if descending or ascending based on angle change
            if knee_angle < self.last_knee_angle - 2:  # Going down
                target_phase = SquatPhase.DESCENDING
            elif knee_angle > self.last_knee_angle + 2:  # Going up
                target_phase = SquatPhase.ASCENDING
            else:
                target_phase = self.current_phase  # No significant change
        else:
            target_phase = self.current_phase
        
        # Update phase with hysteresis (require multiple frames)
        if target_phase != self.current_phase:
            self.phase_frame_count = 1
            previous_phase = self.current_phase
            self.current_phase = target_phase
            
            # Log phase transitions
            logger.debug(f"🔄 Phase transition: {previous_phase.value} → {target_phase.value} (angle: {knee_angle:.1f}°)")
            
            # Count rep when transitioning from ascending to standing
            if previous_phase == SquatPhase.ASCENDING and target_phase == SquatPhase.STANDING:
                self.rep_count += 1
                timestamp = time.time()
                message = f"Rep {self.rep_count} complete! 🎉"
                logger.info(f"🎯 REP COUNTED! #{self.rep_count} | Timestamp: {timestamp:.3f} | Knee angle: {knee_angle:.1f}°")
                logger.info(f"✅ Total reps: {self.rep_count}")
                
                # Invoke callback if provided
                if self.on_squat_complete:
                    try:
                        result = self.on_squat_complete(self.rep_count, knee_angle, timestamp)
                        # If callback is async, schedule it
                        if asyncio.iscoroutine(result):
                            # Try to get the current event loop
                            try:
                                loop = asyncio.get_running_loop()
                                # If loop is running, create a task (fire and forget)
                                asyncio.create_task(result)
                            except RuntimeError:
                                # No running loop, try to get existing loop
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        asyncio.create_task(result)
                                    else:
                                        # Loop exists but not running - schedule it
                                        loop.run_until_complete(result)
                                except RuntimeError:
                                    # No event loop at all, create a new one
                                    # This is a fallback but should rarely happen
                                    asyncio.run(result)
                    except Exception as e:
                        logger.error(f"Error in squat completion callback: {e}", exc_info=True)
        else:
            self.phase_frame_count += 1
        
        self.last_knee_angle = knee_angle
        
        return message
    
    async def process_image(
        self, 
        image: Image.Image, 
        participant: models_pb2.Participant
    ) -> Dict[str, Any]:
        """
        Process image to detect squats.
        This is called after YOLO pose detection.
        """
        # This processor expects to receive pose data from metadata
        # In practice, it should be chained after YOLOPoseProcessor
        return {}
    
    def process_pose_data(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pose data from YOLO to detect squats.
        This should be called by passing YOLO output to this method.
        """
        import time
        
        if not pose_data or "persons" not in pose_data:
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase.value,
                "message": None,
            }
        
        persons = pose_data["persons"]
        if not persons:
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase.value,
                "message": None,
            }
        
        # Use first detected person
        person = persons[0]
        keypoints = person.get("keypoints", [])
        
        if not keypoints:
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase.value,
                "message": None,
            }
        
        # Analyze squat position
        analysis = self._analyze_squat_position(keypoints)
        
        if not analysis.get("valid"):
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase.value,
                "message": None,
            }
        
        knee_angle = analysis["knee_angle"]
        
        # Log current state periodically (every 10 frames)
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        
        if self._frame_count % 10 == 0:
            logger.debug(f"📊 Current state: phase={self.current_phase.value}, knee={knee_angle:.1f}°, reps={self.rep_count}")
        
        # Detect form issues
        self.form_issues = self._detect_form_issues(keypoints, knee_angle)
        if self.form_issues:
            logger.debug(f"⚠️  Form issues detected: {', '.join(self.form_issues)}")
        
        # Update phase and check for rep completion
        message = self._update_phase(knee_angle)
        
        result = {
            "rep_count": self.rep_count,
            "phase": self.current_phase.value,
            "knee_angle": round(knee_angle, 1),
            "hip_angle": round(analysis["hip_angle"], 1) if analysis["hip_angle"] else None,
            "form_issues": self.form_issues,
            "message": message,
            "timestamp": time.time(),
        }
        
        # Log when message is generated (rep counted)
        if message:
            logger.info(f"📢 Message generated: '{message}' at {result['timestamp']:.3f}")
        
        return result
    
    def state(self) -> Dict[str, Any]:
        """Return current state for LLM context."""
        return {
            "rep_count": self.rep_count,
            "current_phase": self.current_phase.value,
            "form_issues": self.form_issues,
            "last_knee_angle": round(self.last_knee_angle, 1) if self.last_knee_angle else None,
        }
    
    def reset(self):
        """Reset counter."""
        self.rep_count = 0
        self.current_phase = SquatPhase.STANDING
        self.phase_frame_count = 0
        self.last_knee_angle = None
        self.form_issues = []
        logger.info("🔄 Squat counter reset")

