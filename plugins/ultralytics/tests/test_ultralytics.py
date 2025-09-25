"""
Tests for the ultralytics plugin.
"""

import pytest
from stream_agents.plugins.ultralytics import YOLOPoseProcessor


class TestYOLOPoseProcessor:
    """Test cases for YOLOPoseProcessor."""

    def test_processor_initialization(self):
        """Test that the processor can be initialized."""
        processor = YOLOPoseProcessor(
            model_path="yolo11n-pose.pt",
            conf_threshold=0.5,
            device="cpu"
        )
        
        assert processor.model_path == "yolo11n-pose.pt"
        assert processor.conf_threshold == 0.5
        assert processor.device == "cpu"
        assert processor.enable_hand_tracking is True
        assert processor.enable_wrist_highlights is True

    def test_processor_state(self):
        """Test that the processor state is correctly returned."""
        processor = YOLOPoseProcessor()
        state = processor.state()
        
        assert "processor_type" in state
        assert "model_path" in state
        assert "confidence_threshold" in state
        assert "device" in state
        assert state["processor_type"] == "YOLO Pose Detection"

    def test_processor_cleanup(self):
        """Test that the processor cleans up properly."""
        processor = YOLOPoseProcessor()
        processor.cleanup()
        
        assert processor._shutdown is True
