import logging
from typing import Optional, Dict, Any, Callable, Union, Awaitable
from PIL import Image
import numpy as np
import cv2

from vision_agents.plugins import ultralytics
from squat_counter_processor import SquatCounterProcessor
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

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
        on_squat_complete: Optional[Union[Callable[[int, float, float], None], Callable[[int, float, float], Awaitable[None]]]] = None,
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
        
        # Initialize squat counter
        self.squat_counter = SquatCounterProcessor(
            min_squat_angle=min_squat_angle,
            max_standing_angle=max_standing_angle,
            confidence_threshold=conf_threshold,
            on_squat_complete=on_squat_complete,
        )
        
        logger.info("💪 Squat YOLO Processor initialized with squat counting")
    
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
        
        # Process squat counting
        squat_data = self.squat_counter.process_pose_data(pose_data)
        
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

