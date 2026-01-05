"""
Tests for the Moondream local processor plugin.

Integration tests require HF_TOKEN environment variable (for gated model access):

    export HF_TOKEN="your-token-here"
    uv run pytest plugins/moondream/tests/test_moondream_local.py -m integration -v

To run only unit tests (no model loading):

    uv run pytest plugins/moondream/tests/test_moondream_local.py -m "not integration" -v
"""

import asyncio
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import torch
from PIL import Image
import av

from vision_agents.plugins.moondream import LocalDetectionProcessor
from vision_agents.plugins.moondream.moondream_utils import annotate_detections
import logging

logger = logging.getLogger(__name__)


@pytest.mark.skip("Skip Moondream local tests because they take too long to run")
class TestMoondreamLocalProcessor:
    """Test cases for MoondreamLocalProcessor."""

    @pytest.fixture(scope="session")
    def golf_image(self, assets_dir) -> Iterator[Image.Image]:
        """Load the local golf swing test image from tests/test_assets."""
        asset_path = Path(assets_dir) / "golf_swing.png"
        with Image.open(asset_path) as img:
            yield img.convert("RGB")

    @pytest.fixture
    def moondream_processor(self) -> Iterator[LocalDetectionProcessor]:
        """Create and manage MoondreamLocalProcessor lifecycle."""
        processor = LocalDetectionProcessor(force_cpu=True)
        try:
            yield processor
        finally:
            processor.close()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN"),
        reason="HF_TOKEN environment variable not set (required for model access)",
    )
    async def test_model_loads_correctly(
        self, moondream_processor: LocalDetectionProcessor
    ):
        """Test that load() successfully loads the model."""
        # Model should be None initially
        assert moondream_processor.model is None

        # Start the processor (loads the model)
        await moondream_processor.warmup()

        # Verify model is loaded
        assert moondream_processor.model is not None
        # Verify model is in eval mode
        assert moondream_processor.model.training is False

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN"),
        reason="HF_TOKEN environment variable not set (required for model access)",
    )
    async def test_run_inference_on_image(
        self, golf_image: Image.Image, moondream_processor: LocalDetectionProcessor
    ):
        """Test _run_inference() with a test image."""
        # Ensure model is loaded
        await moondream_processor.warmup()

        # Convert PIL image to numpy array
        frame_array = np.array(golf_image)

        # Run inference
        result = await moondream_processor._run_inference(frame_array)

        # Verify result structure
        assert isinstance(result, dict)
        assert "detections" in result
        assert isinstance(result["detections"], list)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN"),
        reason="HF_TOKEN environment variable not set (required for model access)",
    )
    async def test_run_detection_sync(
        self, golf_image: Image.Image, moondream_processor: LocalDetectionProcessor
    ):
        """Test _run_detection_sync() directly with PIL Image."""
        # Ensure model is loaded
        await moondream_processor.warmup()

        # Run detection in executor (simulating async context)
        detections = await asyncio.get_event_loop().run_in_executor(
            moondream_processor.executor,
            moondream_processor._run_detection_sync,
            golf_image,
        )

        # Verify return value
        assert isinstance(detections, list)

        # If detections found, verify structure
        if detections:
            for detection in detections:
                assert "label" in detection
                assert "bbox" in detection
                assert "confidence" in detection
                assert isinstance(detection["bbox"], list)
                assert len(detection["bbox"]) == 4

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN"),
        reason="HF_TOKEN environment variable not set (required for model access)",
    )
    async def test_annotated_frame_output(
        self, golf_image: Image.Image, moondream_processor: LocalDetectionProcessor
    ):
        """Test end-to-end frame processing with annotations."""
        # Ensure model is loaded
        await moondream_processor.warmup()

        # Convert PIL Image to av.VideoFrame
        frame = av.VideoFrame.from_image(golf_image)

        # Process the frame
        await moondream_processor._process_and_add_frame(frame)

        # Verify results were stored
        assert hasattr(moondream_processor, "_last_results")
        assert "detections" in moondream_processor._last_results

        # Verify annotated frame was added to video track
        # (We can't easily verify the exact frame without more complex setup,
        # but we can verify the processing didn't fail)

    @pytest.mark.integration
    async def test_annotate_detections_with_results(
        self, golf_image: Image.Image, moondream_processor: LocalDetectionProcessor
    ):
        """Test annotation function directly with mock results."""
        frame_array = np.array(golf_image)

        # Create mock detection results
        mock_results = {
            "detections": [
                {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "person", "confidence": 0.95},
                {"bbox": [100, 200, 300, 400], "label": "car", "confidence": 0.88},
            ]
        }

        annotated = annotate_detections(
            frame_array,
            mock_results,
            font=moondream_processor._font,
            font_scale=moondream_processor._font_scale,
            font_thickness=moondream_processor._font_thickness,
            bbox_color=moondream_processor._bbox_color,
            text_color=moondream_processor._text_color,
        )

        # Verify output shape matches input
        assert annotated.shape == frame_array.shape
        # Verify frame is modified (not array_equal)
        assert not np.array_equal(frame_array, annotated)

    def test_device_auto_detection_cuda(self, monkeypatch):
        """Test CUDA auto-detection."""
        # Mock CUDA available, MPS not available
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        # Ensure torch.backends.mps exists for the test
        if not hasattr(torch.backends, "mps"):
            # Create a mock mps module
            class MockMPS:
                @staticmethod
                def is_available():
                    return False

            monkeypatch.setattr(torch.backends, "mps", MockMPS())
        else:
            monkeypatch.setattr(
                torch.backends.mps,
                "is_available",
                lambda: False,
            )

        # Initialize processor without device param
        processor = LocalDetectionProcessor()
        try:
            assert processor.device == "cuda"
        finally:
            processor.close()

    def test_device_auto_detection_cpu(self, monkeypatch):
        """Test CPU fallback when CUDA and MPS are unavailable."""
        # Mock both CUDA and MPS as unavailable
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Ensure torch.backends.mps exists for the test
        if not hasattr(torch.backends, "mps"):
            # Create a mock mps module
            class MockMPS:
                @staticmethod
                def is_available():
                    return False

            monkeypatch.setattr(torch.backends, "mps", MockMPS())
        else:
            monkeypatch.setattr(
                torch.backends.mps,
                "is_available",
                lambda: False,
            )

        # Initialize processor without device param
        processor = LocalDetectionProcessor()
        try:
            assert processor.device == "cpu"
        finally:
            processor.close()

    def test_device_mps_converted_to_cpu(self, monkeypatch):
        """Test MPS override to CPU (moondream doesn't work with MPS)."""
        # Mock CUDA not available
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Ensure torch.backends.mps exists and mock it as available
        if not hasattr(torch.backends, "mps"):
            # Create a mock mps module with is_available returning True
            class MockMPS:
                @staticmethod
                def is_available():
                    return True

            monkeypatch.setattr(torch.backends, "mps", MockMPS())
        else:
            monkeypatch.setattr(
                torch.backends.mps,
                "is_available",
                lambda: True,
            )

        # Initialize processor - should auto-detect and convert MPS to CPU
        processor = LocalDetectionProcessor()
        try:
            # Verify MPS is converted to CPU
            assert processor.device == "cpu"
        finally:
            processor.close()

        # Also test explicit MPS parameter
        processor2 = LocalDetectionProcessor(force_cpu=True)
        try:
            # Verify explicit MPS is also converted to CPU
            assert processor2.device == "cpu"
        finally:
            processor2.close()

    def test_device_explicit_cpu(self):
        """Test explicit CPU device selection."""
        processor = LocalDetectionProcessor(force_cpu=True)
        try:
            assert processor.device == "cpu"
        finally:
            processor.close()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available on this system",
    )
    def test_device_explicit_cuda(self):
        """Test explicit CUDA device selection (only if CUDA available)."""
        processor = LocalDetectionProcessor()
        try:
            assert processor.device == "cuda"
        finally:
            processor.close()
