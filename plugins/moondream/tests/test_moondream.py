"""
Moondream processor tests.

Unit tests run without API keys.
Integration tests require MOONDREAM_API_KEY environment variable:

    export MOONDREAM_API_KEY="your-key-here"
    uv run pytest plugins/moondream/tests/ -m integration -v

To run only unit tests (no API key needed):

    uv run pytest plugins/moondream/tests/ -m "not integration" -v
"""

import os
import pytest
import av
import numpy as np
from typing import Dict, Any
from PIL import Image

from vision_agents.plugins.moondream import (
    CloudDetectionProcessor,
    MoondreamVideoTrack,
)
from vision_agents.plugins.moondream.moondream_utils import annotate_detections


@pytest.fixture
def sample_image():
    """Test image fixture for Moondream testing."""
    return Image.new("RGB", (640, 480), color="blue")


@pytest.fixture
def sample_frame(sample_image):
    """Test av.VideoFrame fixture."""
    return av.VideoFrame.from_image(sample_image)


async def test_processor_initialization():
    """Test that processor can be initialized with basic config."""
    processor = CloudDetectionProcessor(api_key="test_key")
    assert processor is not None
    await processor.close()


async def test_video_track_frame_queuing(sample_frame):
    """Test that video track can queue and receive frames."""
    track = MoondreamVideoTrack()
    await track.add_frame(sample_frame)
    received_frame = await track.recv()
    assert received_frame is not None
    assert received_frame.width == 640
    assert received_frame.height == 480
    track.stop()


async def test_processor_publishes_track():
    """Test that processor publishes a MoondreamVideoTrack."""
    processor = CloudDetectionProcessor(api_key="test_key")
    track = processor.publish_video_track()
    assert isinstance(track, MoondreamVideoTrack)
    await processor.close()


async def test_cloud_inference_structure(sample_image):
    """Test that cloud inference returns proper structure."""
    processor = CloudDetectionProcessor(api_key="test_key")

    # Mock the SDK detection call
    def mock_detection_sync(image):
        return [{"label": "test", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9}]

    processor._run_detection_sync = mock_detection_sync

    frame_array = np.array(sample_image)
    result = await processor._run_inference(frame_array)

    assert isinstance(result, dict)
    assert "detections" in result
    await processor.close()


async def test_run_inference(sample_image):
    """Test that run_inference works correctly."""
    frame_array = np.array(sample_image)

    # Mock SDK detection
    def mock_detection_sync(image):
        return []

    # Test inference
    processor = CloudDetectionProcessor(api_key="test_key")
    processor._run_detection_sync = mock_detection_sync
    result = await processor._run_inference(frame_array)
    assert isinstance(result, dict)
    await processor.close()


async def test_annotate_detections_with_normalized_coords(sample_image):
    """Test annotation with normalized coordinates."""
    processor = CloudDetectionProcessor(api_key="test_key")

    frame_array = np.array(sample_image)

    # Mock detection results with normalized coordinates
    mock_results = {
        "detections": [
            {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "person", "confidence": 0.95}
        ]
    }

    # Call annotate_detections directly with styling parameters
    annotated = annotate_detections(
        frame_array,
        mock_results,
        font=processor._font,
        font_scale=processor._font_scale,
        font_thickness=processor._font_thickness,
        bbox_color=processor._bbox_color,
        text_color=processor._text_color,
    )

    # Verify frame was modified
    assert not np.array_equal(frame_array, annotated)
    assert annotated.shape == frame_array.shape
    await processor.close()


async def test_annotate_detections_with_pixel_coords(sample_image):
    """Test annotation with pixel coordinates."""
    processor = CloudDetectionProcessor(api_key="test_key")

    frame_array = np.array(sample_image)

    # Mock detection results with pixel coordinates
    mock_results = {
        "detections": [{"bbox": [10, 10, 100, 100], "label": "car", "confidence": 0.88}]
    }

    # Call annotate_detections directly with styling parameters
    annotated = annotate_detections(
        frame_array,
        mock_results,
        font=processor._font,
        font_scale=processor._font_scale,
        font_thickness=processor._font_thickness,
        bbox_color=processor._bbox_color,
        text_color=processor._text_color,
    )

    # Verify frame was modified
    assert not np.array_equal(frame_array, annotated)
    assert annotated.shape == frame_array.shape
    await processor.close()


async def test_annotate_detections_multiple_objects(sample_image):
    """Test annotation with multiple detections."""
    processor = CloudDetectionProcessor(api_key="test_key")

    frame_array = np.array(sample_image)

    # Mock multiple detections
    mock_results = {
        "detections": [
            {"bbox": [0.1, 0.1, 0.3, 0.3], "label": "person", "confidence": 0.95},
            {"bbox": [0.5, 0.5, 0.9, 0.9], "label": "car", "confidence": 0.88},
            {"bbox": [100, 200, 300, 400], "label": "dog", "confidence": 0.92},
        ]
    }

    # Call annotate_detections directly with styling parameters
    annotated = annotate_detections(
        frame_array,
        mock_results,
        font=processor._font,
        font_scale=processor._font_scale,
        font_thickness=processor._font_thickness,
        bbox_color=processor._bbox_color,
        text_color=processor._text_color,
    )

    # Verify frame was modified
    assert not np.array_equal(frame_array, annotated)
    await processor.close()


async def test_annotate_detections_empty_results(sample_image):
    """Test annotation with no detections."""
    processor = CloudDetectionProcessor(api_key="test_key")

    frame_array = np.array(sample_image)
    mock_results: Dict[str, Any] = {"detections": []}

    # Call annotate_detections directly with styling parameters
    annotated = annotate_detections(
        frame_array,
        mock_results,
        font=processor._font,
        font_scale=processor._font_scale,
        font_thickness=processor._font_thickness,
        bbox_color=processor._bbox_color,
        text_color=processor._text_color,
    )

    # Frame should be unchanged
    assert np.array_equal(frame_array, annotated)
    await processor.close()


async def test_process_and_add_frame(sample_frame):
    """Test the full frame processing pipeline."""
    processor = CloudDetectionProcessor(api_key="test_key")

    # Mock the run_inference method to return test data
    async def mock_inference(frame_array):
        return {
            "detections": [
                {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "test", "confidence": 0.9}
            ]
        }

    processor.run_inference = mock_inference

    # Process a frame
    await processor._process_and_add_frame(sample_frame)

    # Verify results were stored
    assert hasattr(processor, "_last_results")
    assert "detections" in processor._last_results
    await processor.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set"
)
async def test_live_detection_api():
    """Test live detection API with real Moondream service."""
    processor = CloudDetectionProcessor(
        api_key=os.getenv("MOONDREAM_API_KEY"), conf_threshold=0.5
    )

    # Use existing test image
    from pathlib import Path

    assets_dir = Path(__file__).parent.parent.parent.parent / "tests" / "test_assets"
    image_path = assets_dir / "golf_swing.png"

    if image_path.exists():
        image = Image.open(image_path)
        frame_array = np.array(image)

        # Run inference
        result = await processor._run_inference(frame_array)

        # Verify we got real detections
        assert "detections" in result
        assert isinstance(result["detections"], list)

        # Log what we detected
        if result["detections"]:
            print(f"\n✅ Detected {len(result['detections'])} objects:")
            for det in result["detections"]:
                print(f"  - {det.get('label')} ({det.get('confidence', 0):.2f})")
        else:
            print("\n⚠️ No objects detected (this might be expected)")
    else:
        pytest.skip(f"Test image not found: {image_path}")

    await processor.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set"
)
async def test_live_detection_with_annotation():
    """Test that detection results are properly annotated on frames."""
    processor = CloudDetectionProcessor(api_key=os.getenv("MOONDREAM_API_KEY"))

    # Create a simple test image
    test_image = Image.new("RGB", (640, 480), color="blue")
    frame_array = np.array(test_image)

    # Run inference
    result = await processor._run_inference(frame_array)

    # If we got detections, test annotation
    if result.get("detections"):
        # Call annotate_detections directly with styling parameters
        annotated = annotate_detections(
            frame_array,
            result,
            font=processor._font,
            font_scale=processor._font_scale,
            font_thickness=processor._font_thickness,
            bbox_color=processor._bbox_color,
            text_color=processor._text_color,
        )

        # Verify frame was modified
        assert not np.array_equal(frame_array, annotated)

        # Optionally save for visual inspection
        # Image.fromarray(annotated).save("/tmp/moondream_test_annotated.jpg")

    await processor.close()


def test_missing_api_key(monkeypatch):
    """Test that missing API key raises ValueError when env var is also missing."""
    # Remove the environment variable to test the error case
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="api_key is required"):
        CloudDetectionProcessor(api_key=None)


async def test_api_key_from_env(monkeypatch):
    """Test that API key is loaded from environment variable."""
    monkeypatch.setenv("MOONDREAM_API_KEY", "test_env_key")

    processor = CloudDetectionProcessor()
    assert processor.api_key == "test_env_key"
    await processor.close()


async def test_api_key_explicit_override(monkeypatch):
    """Test that explicit API key overrides environment variable."""
    monkeypatch.setenv("MOONDREAM_API_KEY", "env_key")

    processor = CloudDetectionProcessor(api_key="explicit_key")
    assert processor.api_key == "explicit_key"
    await processor.close()


async def test_detect_objects_default():
    """Test default detect_objects is 'person'."""
    processor = CloudDetectionProcessor(api_key="test_key")
    assert processor.detect_objects == ["person"]
    await processor.close()


async def test_detect_objects_single_string():
    """Test detect_objects with single string."""
    processor = CloudDetectionProcessor(api_key="test_key", detect_objects="car")
    assert processor.detect_objects == ["car"]
    await processor.close()


async def test_detect_objects_list():
    """Test detect_objects with list."""
    processor = CloudDetectionProcessor(
        api_key="test_key", detect_objects=["person", "car", "dog"]
    )
    assert processor.detect_objects == ["person", "car", "dog"]
    await processor.close()


def test_detect_objects_invalid_type():
    """Test detect_objects with invalid type raises error."""
    with pytest.raises(ValueError, match="detect_objects must be str or list"):
        CloudDetectionProcessor(
            api_key="test_key",
            detect_objects=123,  # Invalid: not a string or list
        )


def test_detect_objects_invalid_list_contents():
    """Test detect_objects with non-string list items raises error."""
    with pytest.raises(ValueError, match="detect_objects must be str or list"):
        CloudDetectionProcessor(
            api_key="test_key",
            detect_objects=["person", 123, "car"],  # Invalid: contains non-string
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set"
)
async def test_custom_object_detection():
    """Test detection with custom object type (not 'person')."""
    processor = CloudDetectionProcessor(
        api_key=os.getenv("MOONDREAM_API_KEY"),
        detect_objects="car",  # Detect cars instead of persons
    )

    # Use golf_swing.png - might not have cars, but test should run
    from pathlib import Path

    assets_dir = Path(__file__).parent.parent.parent.parent / "tests" / "test_assets"
    image_path = assets_dir / "golf_swing.png"

    if image_path.exists():
        image = Image.open(image_path)
        frame_array = np.array(image)

        # Run inference - may return empty if no cars in image
        result = await processor._run_inference(frame_array)

        # Verify structure is correct
        assert "detections" in result
        assert isinstance(result["detections"], list)

        # If any detections, verify label is "car"
        for det in result.get("detections", []):
            assert det["label"] == "car", f"Expected 'car' but got '{det['label']}'"

        print(
            f"\n🚗 Car detection test: Found {len(result.get('detections', []))} cars"
        )
    else:
        pytest.skip(f"Test image not found: {image_path}")

    await processor.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set"
)
async def test_multiple_object_detection():
    """Test detection with multiple object types."""
    processor = CloudDetectionProcessor(
        api_key=os.getenv("MOONDREAM_API_KEY"),
        detect_objects=["person", "grass", "sky"],  # Multiple types
    )

    # Use golf_swing.png - likely has person and grass
    from pathlib import Path

    assets_dir = Path(__file__).parent.parent.parent.parent / "tests" / "test_assets"
    image_path = assets_dir / "golf_swing.png"

    if image_path.exists():
        image = Image.open(image_path)
        frame_array = np.array(image)

        # Run inference
        result = await processor._run_inference(frame_array)

        # Verify structure
        assert "detections" in result
        assert isinstance(result["detections"], list)

        # Log what was detected
        detected_labels = [det["label"] for det in result.get("detections", [])]
        unique_labels = set(detected_labels)

        print("\n🎯 Multiple object detection test:")
        print(f"   Searched for: {processor.detect_objects}")
        print(f"   Found {len(result.get('detections', []))} total detections")
        print(f"   Unique object types: {unique_labels}")

        # Verify all labels are from our configured list
        for label in detected_labels:
            assert label in processor.detect_objects, (
                f"Detected '{label}' but it's not in configured objects {processor.detect_objects}"
            )
    else:
        pytest.skip(f"Test image not found: {image_path}")

    await processor.close()
