"""
Tests for the ultralytics plugin.
"""

import asyncio
from pathlib import Path
from typing import Iterator

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

    async def test_pose_data_from_image(self, golf_image: Image.Image, pose_processor: YOLOPoseProcessor):
        """Run pose detection on the golf swing image and validate pose data is returned."""
        result = await pose_processor.process_image(image=golf_image, user_id="ultra-test")

        assert result is not None
        assert isinstance(result, dict)
        assert "pose_data" in result
        assert isinstance(result["pose_data"], dict)
        # pose_data may be empty on certain images, but structure must exist
        assert "persons" in result["pose_data"]

    async def test_annotated_image_output(self, golf_image: Image.Image, pose_processor: YOLOPoseProcessor):
        """Run pose detection and ensure an annotated image is produced."""
        result = await pose_processor.process_image(image=golf_image, user_id="ultra-test")

        assert result is not None
        assert "annotated_image" in result
        logger.info(result)
        print(result.get("pose_data"))
        annotated_image = result["annotated_image"]
        assert isinstance(annotated_image, Image.Image)
        # Ensure same size as input for simplicity
        assert annotated_image.size == golf_image.size
        
        # Save the annotated image temporarily for inspection
        temp_path = Path("/tmp/annotated_golf_swing.png")
        annotated_image.save(temp_path)
        print(f"Saved annotated image to: {temp_path}")


