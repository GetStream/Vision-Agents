"""
Tests for the ultralytics plugin.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from PIL import Image
import av

from vision_agents.plugins.ultralytics import YOLOPoseProcessor
import logging

logger = logging.getLogger(__name__)


class TestYOLOPoseProcessor:
    """Test cases for YOLOPoseProcessor."""

    @pytest.fixture(scope="session")
    def golf_image(self, assets_dir) -> Iterator[Image.Image]:
        """Load the local golf swing test image from tests/test_assets."""
        asset_path = Path(assets_dir) / "golf_swing.png"
        with Image.open(asset_path) as img:
            yield img.convert("RGB")

    @pytest.fixture
    def pose_processor(self) -> Iterator[YOLOPoseProcessor]:
        """Create and manage YOLOPoseProcessor lifecycle."""
        processor = YOLOPoseProcessor(device="cpu")
        try:
            yield processor
        finally:
            processor.close()

    async def test_annotated_ndarray(self, golf_image: Image.Image, pose_processor: YOLOPoseProcessor):
        frame_array = np.array(golf_image)
        array_with_pose, pose = await pose_processor.add_pose_to_ndarray(frame_array)

        assert array_with_pose is not None
        assert pose is not None

    async def test_annotated_image_output(self, golf_image: Image.Image, pose_processor: YOLOPoseProcessor):
        image_with_pose, pose = await pose_processor.add_pose_to_image(image=golf_image)

        assert image_with_pose is not None
        assert pose is not None

        # Ensure same size as input for simplicity
        assert image_with_pose.size == golf_image.size
        
        # Save the annotated image temporarily for inspection
        temp_path = Path("/tmp/annotated_golf_swing.png")
        image_with_pose.save(temp_path)
        print(f"Saved annotated image to: {temp_path}")

    async def test_annotated_frame_output(self, golf_image: Image.Image, pose_processor: YOLOPoseProcessor):
        """Test add_pose_to_frame method with av.VideoFrame input."""
        # Convert PIL Image to av.VideoFrame
        frame = av.VideoFrame.from_image(golf_image)
        
        # Process the frame with pose detection
        frame_with_pose = await pose_processor.add_pose_to_frame(frame)
        
        # Verify the result is an av.VideoFrame
        assert frame_with_pose is not None
        assert isinstance(frame_with_pose, av.VideoFrame)
        
        # Verify dimensions are preserved
        assert frame_with_pose.width == frame.width
        assert frame_with_pose.height == frame.height
        
        # Convert back to numpy array to verify it's been processed
        result_array = frame_with_pose.to_ndarray()
        original_array = frame.to_ndarray()
        
        # The arrays should be the same shape
        assert result_array.shape == original_array.shape
        
        # The processed frame should be different from the original (pose annotations added)
        # Note: This might not always be true if no pose is detected, but it's a reasonable check
        assert not np.array_equal(result_array, original_array)


