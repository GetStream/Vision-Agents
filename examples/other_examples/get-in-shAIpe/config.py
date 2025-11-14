"""
Configuration file for the squat counter.

Adjust these values to tune the squat detection for your needs.
"""

# Squat Detection Parameters
SQUAT_CONFIG = {
    # Angle thresholds (in degrees)
    "min_squat_angle": 100.0,  # Minimum knee angle at bottom (lower = deeper squat required)
    "max_standing_angle": 160.0,  # Maximum knee angle when standing (higher = more upright)
    
    # Detection confidence
    "confidence_threshold": 0.5,  # Minimum confidence for keypoint detection (0.0-1.0)
    
    # Performance settings
    "enable_hand_tracking": False,  # Disable for better performance
    "enable_wrist_highlights": False,  # Disable for better performance
}

# YOLO Model Settings
YOLO_CONFIG = {
    "model_path": "yolo11n-pose.pt",  # Path to YOLO pose model
    "imgsz": 512,  # Image size for inference
    "device": "cpu",  # "cpu" or "cuda"
    "max_workers": 24,  # Number of worker threads
    "fps": 30,  # Target FPS
    "interval": 0,  # Processing interval (0 = every frame)
}

# LLM Settings
LLM_CONFIG = {
    "provider": "gemini",  # "gemini" or "openai"
    "fps": 3,  # Frames per second sent to LLM (lower = cheaper)
}

# Form Detection Thresholds
FORM_CONFIG = {
    "shallow_squat_threshold": 110,  # Knee angle above this at bottom = too shallow
    "knee_forward_threshold": 50,  # Pixels knee is forward of ankle = bad form
}

# Visual Overlay Settings
VISUAL_CONFIG = {
    "show_rep_count": True,
    "show_phase": True,
    "show_angles": True,
    "show_form_issues": True,
    "show_messages": True,
}

# Coaching Settings
COACHING_CONFIG = {
    "milestone_reps": [5, 10, 15, 20],  # Rep counts to celebrate
    "provide_phase_coaching": True,  # Give tips during different phases
    "provide_form_corrections": True,  # Alert on form issues
}

