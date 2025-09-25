"""
Tests for the ultralytics plugin.
"""

import asyncio
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from PIL import Image

from stream_agents.plugins.ultralytics import YOLOPoseProcessor
from tests.base_test import BaseTest
import logging

logger = logging.getLogger(__name__)

class TestYOLOPoseProcessor(BaseTest):
    """Test cases for YOLOPoseProcessor."""

    @pytest.fixture(scope="session")
    def golf_image(self) -> Iterator[Image.Image]:
        """Load the local golf swing test image from tests/test_assets."""
        asset_path = Path(self.assets_dir) / "golf_swing.png"
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


