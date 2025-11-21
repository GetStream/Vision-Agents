"""Tests for SAM3 video segmentation processor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import av
import numpy as np
import pytest

from vision_agents.plugins.sam3 import VideoSegmentationProcessor


class TestVideoSegmentationProcessor:
    """Tests for the SAM3 VideoSegmentationProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        proc = VideoSegmentationProcessor(
            text_prompt="person",
            fps=30,
            force_cpu=True,  # Force CPU for tests
        )
        yield proc
        proc.close()

    def test_initialization(self, processor):
        """Test processor initializes correctly."""
        assert processor.text_prompt == "person"
        assert processor.fps == 30
        assert processor.name == "sam3_video_segmentation"
        assert processor.device == "cpu"

    async def test_change_prompt(self, processor):
        """Test changing the segmentation prompt."""
        result = await processor.change_prompt("car")
        
        assert result["status"] == "success"
        assert processor.text_prompt == "car"
        assert "person" in result["message"]
        assert "car" in result["message"]

    async def test_change_prompt_updates_text(self, processor):
        """Test that changing prompt updates the text prompt."""
        initial_prompt = processor.text_prompt
        
        await processor.change_prompt("dog")
        
        assert processor.text_prompt == "dog"
        assert processor.text_prompt != initial_prompt

    def test_annotate_segmentation_empty_results(self, processor):
        """Test annotation with empty results."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {}
        
        annotated = processor._annotate_segmentation(frame, results)
        
        # Should return frame unchanged
        assert annotated.shape == frame.shape
        assert np.array_equal(annotated, frame)

    def test_annotate_segmentation_with_boxes(self, processor):
        """Test annotation with bounding boxes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {
            "boxes": [[10, 10, 100, 100], [200, 200, 300, 300]],
            "scores": [0.95, 0.87]
        }
        
        annotated = processor._annotate_segmentation(frame, results)
        
        # Should have modified the frame
        assert annotated.shape == frame.shape
        # Frame should have some non-zero values from drawing
        assert not np.array_equal(annotated, frame)

    def test_device_property(self, processor):
        """Test device property returns string."""
        assert isinstance(processor.device, str)
        assert processor.device == "cpu"

    def test_publish_video_track(self, processor):
        """Test video track publishing."""
        track = processor.publish_video_track()
        
        assert track is not None
        assert hasattr(track, "add_frame")
        assert hasattr(track, "recv")

    def test_close(self, processor):
        """Test processor cleanup."""
        processor.model = MagicMock()
        processor.processor = MagicMock()
        
        processor.close()
        
        assert processor._shutdown is True
        assert processor.model is None
        assert processor.processor is None

    @pytest.mark.integration
    async def test_warmup_with_mock(self, processor):
        """Test warmup loads the model."""
        # Mock the model and processor loading
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        with patch.object(processor, "_load_sam3_sync", return_value=(mock_model, mock_processor)):
            await processor.warmup()
            
            assert processor.model is not None
            assert processor.processor is not None

    async def test_process_and_add_frame_error_handling(self, processor):
        """Test frame processing handles errors gracefully."""
        # Create a mock frame
        mock_frame = MagicMock(spec=av.VideoFrame)
        frame_array = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_frame.to_ndarray.return_value = frame_array
        
        # Mock video track
        processor._video_track = AsyncMock()
        processor._video_track.add_frame = AsyncMock()
        
        # Mock segmentation to raise an error
        with patch.object(processor, "_run_segmentation", side_effect=Exception("Test error")):
            await processor._process_and_add_frame(mock_frame)
            
            # Should still add a frame (the original one)
            assert processor._video_track.add_frame.called

