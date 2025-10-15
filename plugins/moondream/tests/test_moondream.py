"""
Moondream processor tests.

Unit tests run without API keys.
Integration tests require MOONDREAM_API_KEY environment variable:

    export MOONDREAM_API_KEY="your-key-here"
    pytest plugins/moondream/tests/ -m integration -v
    
To run only unit tests (no API key needed):

    pytest plugins/moondream/tests/ -m "not integration" -v
"""
import os
import pytest
import av
import numpy as np
from PIL import Image

from vision_agents.plugins.moondream import (
    MoondreamProcessor,
    MoondreamVideoTrack,
    MoondreamAPIError,
    MoondreamAuthError,
    MoondreamRateLimitError,
    MoondreamBadRequestError,
)


@pytest.fixture
def sample_image():
    """Test image fixture for Moondream testing."""
    return Image.new("RGB", (640, 480), color="blue")


@pytest.fixture
def sample_frame(sample_image):
    """Test av.VideoFrame fixture."""
    return av.VideoFrame.from_image(sample_image)


def test_processor_initialization():
    """Test that processor can be initialized with basic config."""
    processor = MoondreamProcessor(mode="cloud", api_key="test_key")
    assert processor is not None
    assert processor.mode == "cloud"
    assert processor.skills == ["detection"]
    processor.close()


def test_processor_with_custom_skills():
    """Test processor initialization with custom skills."""
    processor = MoondreamProcessor(
        mode="local",
        skills=["detection", "vqa", "caption"],
        vqa_prompt="What objects are present?"
    )
    assert processor.mode == "local"
    assert "detection" in processor.skills
    assert "vqa" in processor.skills
    assert "caption" in processor.skills
    assert processor.vqa_prompt == "What objects are present?"
    processor.close()


def test_processor_state():
    """Test that processor returns empty state initially."""
    processor = MoondreamProcessor(mode="cloud", api_key="test_key")
    state = processor.state()
    assert isinstance(state, dict)
    processor.close()


# Phase 2 Tests: Video Track + Frame Queuing

@pytest.mark.asyncio
async def test_video_track_frame_queuing(sample_frame):
    """Test that video track can queue and receive frames."""
    track = MoondreamVideoTrack()
    await track.add_frame(sample_frame)
    received_frame = await track.recv()
    assert received_frame is not None
    assert received_frame.width == 640
    assert received_frame.height == 480
    track.stop()


def test_processor_publishes_track():
    """Test that processor publishes a MoondreamVideoTrack."""
    processor = MoondreamProcessor(mode="local")  # local mode doesn't require API key
    track = processor.publish_video_track()
    assert isinstance(track, MoondreamVideoTrack)
    processor.close()


@pytest.mark.asyncio
async def test_video_track_multiple_frames(sample_image):
    """Test that video track handles multiple frames correctly."""
    track = MoondreamVideoTrack()
    
    # Add multiple frames
    for i in range(5):
        img = Image.new("RGB", (640, 480), color=(i*50, 0, 0))
        frame = av.VideoFrame.from_image(img)
        await track.add_frame(frame)
    
    # Receive a frame
    received_frame = await track.recv()
    assert received_frame is not None
    track.stop()


# Phase 3 Tests: Inference Backends

@pytest.mark.asyncio
async def test_cloud_inference_structure(sample_image):
    """Test that cloud inference returns proper structure."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"], api_key="test_key")
    
    # Mock the API call to avoid making a real HTTP request
    async def mock_detection_api(session, img_base64):
        return [{"label": "test", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9}]
    
    processor._call_detection_api = mock_detection_api
    
    frame_array = np.array(sample_image)
    result = await processor._cloud_inference(frame_array)
    
    assert isinstance(result, dict)
    assert "detections" in result
    processor.close()


@pytest.mark.asyncio
async def test_local_inference_structure(sample_image):
    """Test that local inference returns proper structure."""
    processor = MoondreamProcessor(mode="local", skills=["detection"])
    
    frame_array = np.array(sample_image)
    result = await processor._local_inference(frame_array)
    
    assert isinstance(result, dict)
    processor.close()


@pytest.mark.asyncio
async def test_fal_inference_structure(sample_image):
    """Test that FAL inference returns proper structure."""
    processor = MoondreamProcessor(mode="fal", skills=["detection"])
    
    frame_array = np.array(sample_image)
    result = await processor._fal_inference(frame_array)
    
    assert isinstance(result, dict)
    processor.close()


@pytest.mark.asyncio
async def test_run_inference_routing(sample_image):
    """Test that run_inference routes to correct backend."""
    frame_array = np.array(sample_image)
    
    # Test cloud mode
    processor_cloud = MoondreamProcessor(mode="cloud", api_key="test_key")
    result = await processor_cloud.run_inference(frame_array)
    assert isinstance(result, dict)
    processor_cloud.close()
    
    # Test local mode
    processor_local = MoondreamProcessor(mode="local")
    result = await processor_local.run_inference(frame_array)
    assert isinstance(result, dict)
    processor_local.close()
    
    # Test fal mode
    processor_fal = MoondreamProcessor(mode="fal")
    result = await processor_fal.run_inference(frame_array)
    assert isinstance(result, dict)
    processor_fal.close()


@pytest.mark.asyncio
async def test_multi_skill_inference(sample_image):
    """Test inference with multiple skills enabled."""
    processor = MoondreamProcessor(
        mode="cloud",
        skills=["detection", "vqa", "caption"],
        api_key="test_key"
    )
    
    # Mock the API calls
    async def mock_detection_api(session, img_base64):
        return [{"label": "test", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9}]
    
    async def mock_vqa_api(session, img_base64, question):
        return "Mock VQA response"
    
    async def mock_caption_api(session, img_base64):
        return "Mock caption"
    
    processor._call_detection_api = mock_detection_api
    processor._call_vqa_api = mock_vqa_api
    processor._call_caption_api = mock_caption_api
    
    frame_array = np.array(sample_image)
    result = await processor.run_inference(frame_array)
    
    assert isinstance(result, dict)
    # Should have results for all enabled skills
    assert "detections" in result
    assert "vqa_response" in result
    assert "caption" in result
    processor.close()


# Phase 4 Tests: Frame Processing + Annotation

def test_annotate_detections_with_normalized_coords(sample_image):
    """Test annotation with normalized coordinates."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"], api_key="test_key")
    
    frame_array = np.array(sample_image)
    
    # Mock detection results with normalized coordinates
    mock_results = {
        "detections": [
            {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "person", "confidence": 0.95}
        ]
    }
    
    annotated = processor._annotate_detections(frame_array, mock_results)
    
    # Verify frame was modified
    assert not np.array_equal(frame_array, annotated)
    assert annotated.shape == frame_array.shape
    processor.close()


def test_annotate_detections_with_pixel_coords(sample_image):
    """Test annotation with pixel coordinates."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"], api_key="test_key")
    
    frame_array = np.array(sample_image)
    
    # Mock detection results with pixel coordinates
    mock_results = {
        "detections": [
            {"bbox": [10, 10, 100, 100], "label": "car", "confidence": 0.88}
        ]
    }
    
    annotated = processor._annotate_detections(frame_array, mock_results)
    
    # Verify frame was modified
    assert not np.array_equal(frame_array, annotated)
    assert annotated.shape == frame_array.shape
    processor.close()


def test_annotate_detections_multiple_objects(sample_image):
    """Test annotation with multiple detections."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"], api_key="test_key")
    
    frame_array = np.array(sample_image)
    
    # Mock multiple detections
    mock_results = {
        "detections": [
            {"bbox": [0.1, 0.1, 0.3, 0.3], "label": "person", "confidence": 0.95},
            {"bbox": [0.5, 0.5, 0.9, 0.9], "label": "car", "confidence": 0.88},
            {"bbox": [100, 200, 300, 400], "label": "dog", "confidence": 0.92},
        ]
    }
    
    annotated = processor._annotate_detections(frame_array, mock_results)
    
    # Verify frame was modified
    assert not np.array_equal(frame_array, annotated)
    processor.close()


def test_annotate_detections_empty_results(sample_image):
    """Test annotation with no detections."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"], api_key="test_key")
    
    frame_array = np.array(sample_image)
    mock_results = {"detections": []}
    
    annotated = processor._annotate_detections(frame_array, mock_results)
    
    # Frame should be unchanged
    assert np.array_equal(frame_array, annotated)
    processor.close()


@pytest.mark.asyncio
async def test_process_and_add_frame(sample_frame):
    """Test the full frame processing pipeline."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"], api_key="test_key")
    
    # Mock the run_inference method to return test data
    async def mock_inference(frame_array):
        return {"detections": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "test", "confidence": 0.9}]}
    
    processor.run_inference = mock_inference
    
    # Process a frame
    await processor._process_and_add_frame(sample_frame)
    
    # Verify results were stored
    assert hasattr(processor, "_last_results")
    assert "detections" in processor._last_results
    processor.close()


# Phase 6 Tests: State Management + LLM Integration

def test_state_exposes_results():
    """Test that state() provides LLM-friendly data."""
    processor = MoondreamProcessor(mode="cloud", skills=["vqa", "detection"], api_key="test_key")
    
    # Simulate inference results
    processor._last_results = {
        "vqa_response": "There is a person standing",
        "detections": [{"label": "person", "confidence": 0.95, "bbox": [0.1, 0.1, 0.5, 0.5]}]
    }
    processor._last_frame_time = 123.45
    
    state = processor.state()
    assert "vqa_response" in state
    assert state["vqa_response"] == "There is a person standing"
    assert "detections_summary" in state
    assert "person" in state["detections_summary"]
    assert "detections_count" in state
    assert state["detections_count"] == 1
    processor.close()


def test_state_with_caption():
    """Test state with caption skill."""
    processor = MoondreamProcessor(mode="cloud", skills=["caption"], api_key="test_key")
    
    processor._last_results = {"caption": "A sunny day at the beach"}
    
    state = processor.state()
    assert "caption" in state
    assert state["caption"] == "A sunny day at the beach"
    processor.close()


def test_state_with_counting():
    """Test state with counting skill."""
    processor = MoondreamProcessor(mode="cloud", skills=["counting"], api_key="test_key")
    
    processor._last_results = {"count": 5}
    
    state = processor.state()
    assert "count" in state
    assert state["count"] == 5
    processor.close()


def test_state_empty_before_processing():
    """Test that state is empty before any processing."""
    processor = MoondreamProcessor(mode="cloud", api_key="test_key")
    state = processor.state()
    assert state == {}
    processor.close()


def test_summarize_detections():
    """Test detection summarization for LLM."""
    processor = MoondreamProcessor(mode="cloud", api_key="test_key")
    
    detections = [
        {"label": "person", "confidence": 0.95},
        {"label": "person", "confidence": 0.88},
        {"label": "car", "confidence": 0.92},
    ]
    
    summary = processor._summarize_detections(detections)
    assert "2 persons" in summary
    assert "1 car" in summary
    processor.close()


def test_summarize_empty_detections():
    """Test detection summary with no detections."""
    processor = MoondreamProcessor(mode="cloud", api_key="test_key")
    summary = processor._summarize_detections([])
    assert summary == "No objects detected"
    processor.close()


# Phase 7 Tests: Live API Integration

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set")
@pytest.mark.asyncio
async def test_live_detection_api():
    """Test live detection API with real Moondream service."""
    processor = MoondreamProcessor(
        mode="cloud",
        api_key=os.getenv("MOONDREAM_API_KEY"),
        skills=["detection"],
        conf_threshold=0.5
    )
    
    # Use existing test image
    from pathlib import Path
    assets_dir = Path(__file__).parent.parent.parent.parent / "tests" / "test_assets"
    image_path = assets_dir / "golf_swing.png"
    
    if image_path.exists():
        image = Image.open(image_path)
        frame_array = np.array(image)
        
        # Run inference
        result = await processor.run_inference(frame_array)
        
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
    
    processor.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set")
@pytest.mark.asyncio
async def test_live_detection_with_annotation():
    """Test that detection results are properly annotated on frames."""
    processor = MoondreamProcessor(
        mode="cloud",
        api_key=os.getenv("MOONDREAM_API_KEY"),
        skills=["detection"]
    )
    
    # Create a simple test image
    test_image = Image.new("RGB", (640, 480), color="blue")
    frame_array = np.array(test_image)
    
    # Run inference
    result = await processor.run_inference(frame_array)
    
    # If we got detections, test annotation
    if result.get("detections"):
        annotated = processor._annotate_detections(frame_array, result)
        
        # Verify frame was modified
        assert not np.array_equal(frame_array, annotated)
        
        # Optionally save for visual inspection
        # Image.fromarray(annotated).save("/tmp/moondream_test_annotated.jpg")
    
    processor.close()


def test_missing_api_key(monkeypatch):
    """Test that missing API key raises ValueError when env var is also missing."""
    # Remove the environment variable to test the error case
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)
    
    with pytest.raises(ValueError, match="api_key is required"):
        MoondreamProcessor(mode="cloud", api_key=None)


def test_api_key_from_env(monkeypatch):
    """Test that API key is loaded from environment variable."""
    monkeypatch.setenv("MOONDREAM_API_KEY", "test_env_key")
    
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    assert processor.api_key == "test_env_key"
    processor.close()


def test_api_key_explicit_override(monkeypatch):
    """Test that explicit API key overrides environment variable."""
    monkeypatch.setenv("MOONDREAM_API_KEY", "env_key")
    
    processor = MoondreamProcessor(mode="cloud", api_key="explicit_key", skills=["detection"])
    assert processor.api_key == "explicit_key"
    processor.close()


# Phase 8 Tests: Configurable Detection Objects

def test_detect_objects_default():
    """Test default detect_objects is 'person'."""
    processor = MoondreamProcessor(mode="local")
    assert processor.detect_objects == ["person"]
    processor.close()


def test_detect_objects_single_string():
    """Test detect_objects with single string."""
    processor = MoondreamProcessor(
        mode="local",
        detect_objects="car"
    )
    assert processor.detect_objects == ["car"]
    processor.close()


def test_detect_objects_list():
    """Test detect_objects with list."""
    processor = MoondreamProcessor(
        mode="local",
        detect_objects=["person", "car", "dog"]
    )
    assert processor.detect_objects == ["person", "car", "dog"]
    processor.close()


def test_detect_objects_invalid_type():
    """Test detect_objects with invalid type raises error."""
    with pytest.raises(ValueError, match="detect_objects must be str or list"):
        MoondreamProcessor(
            mode="local",
            detect_objects=123  # Invalid: not a string or list
        )


def test_detect_objects_invalid_list_contents():
    """Test detect_objects with non-string list items raises error."""
    with pytest.raises(ValueError, match="detect_objects must be str or list"):
        MoondreamProcessor(
            mode="local",
            detect_objects=["person", 123, "car"]  # Invalid: contains non-string
        )


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set")
@pytest.mark.asyncio
async def test_custom_object_detection():
    """Test detection with custom object type (not 'person')."""
    processor = MoondreamProcessor(
        mode="cloud",
        api_key=os.getenv("MOONDREAM_API_KEY"),
        skills=["detection"],
        detect_objects="car"  # Detect cars instead of persons
    )
    
    # Use golf_swing.png - might not have cars, but test should run
    from pathlib import Path
    assets_dir = Path(__file__).parent.parent.parent.parent / "tests" / "test_assets"
    image_path = assets_dir / "golf_swing.png"
    
    if image_path.exists():
        image = Image.open(image_path)
        frame_array = np.array(image)
        
        # Run inference - may return empty if no cars in image
        result = await processor.run_inference(frame_array)
        
        # Verify structure is correct
        assert "detections" in result
        assert isinstance(result["detections"], list)
        
        # If any detections, verify label is "car"
        for det in result.get("detections", []):
            assert det["label"] == "car", f"Expected 'car' but got '{det['label']}'"
        
        print(f"\n🚗 Car detection test: Found {len(result.get('detections', []))} cars")
    else:
        pytest.skip(f"Test image not found: {image_path}")
    
    processor.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set")
@pytest.mark.asyncio
async def test_multiple_object_detection():
    """Test detection with multiple object types."""
    processor = MoondreamProcessor(
        mode="cloud",
        api_key=os.getenv("MOONDREAM_API_KEY"),
        skills=["detection"],
        detect_objects=["person", "grass", "sky"]  # Multiple types
    )
    
    # Use golf_swing.png - likely has person and grass
    from pathlib import Path
    assets_dir = Path(__file__).parent.parent.parent.parent / "tests" / "test_assets"
    image_path = assets_dir / "golf_swing.png"
    
    if image_path.exists():
        image = Image.open(image_path)
        frame_array = np.array(image)
        
        # Run inference
        result = await processor.run_inference(frame_array)
        
        # Verify structure
        assert "detections" in result
        assert isinstance(result["detections"], list)
        
        # Log what was detected
        detected_labels = [det["label"] for det in result.get("detections", [])]
        unique_labels = set(detected_labels)
        
        print(f"\n🎯 Multiple object detection test:")
        print(f"   Searched for: {processor.detect_objects}")
        print(f"   Found {len(result.get('detections', []))} total detections")
        print(f"   Unique object types: {unique_labels}")
        
        # Verify all labels are from our configured list
        for label in detected_labels:
            assert label in processor.detect_objects, \
                f"Detected '{label}' but it's not in configured objects {processor.detect_objects}"
    else:
        pytest.skip(f"Test image not found: {image_path}")
    
    processor.close()

