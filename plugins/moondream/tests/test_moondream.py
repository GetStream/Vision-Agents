import pytest
import av
import numpy as np
from PIL import Image

from vision_agents.plugins.moondream import MoondreamProcessor, MoondreamVideoTrack


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
    processor = MoondreamProcessor(mode="cloud")
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
    processor = MoondreamProcessor(mode="cloud")
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
    processor = MoondreamProcessor()
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
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    
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
    processor_cloud = MoondreamProcessor(mode="cloud")
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
        skills=["detection", "vqa", "caption"]
    )
    
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
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    
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
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    
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
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    
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
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    
    frame_array = np.array(sample_image)
    mock_results = {"detections": []}
    
    annotated = processor._annotate_detections(frame_array, mock_results)
    
    # Frame should be unchanged
    assert np.array_equal(frame_array, annotated)
    processor.close()


@pytest.mark.asyncio
async def test_process_and_add_frame(sample_frame):
    """Test the full frame processing pipeline."""
    processor = MoondreamProcessor(mode="cloud", skills=["detection"])
    
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
    processor = MoondreamProcessor(mode="cloud", skills=["vqa", "detection"])
    
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
    processor = MoondreamProcessor(mode="cloud", skills=["caption"])
    
    processor._last_results = {"caption": "A sunny day at the beach"}
    
    state = processor.state()
    assert "caption" in state
    assert state["caption"] == "A sunny day at the beach"
    processor.close()


def test_state_with_counting():
    """Test state with counting skill."""
    processor = MoondreamProcessor(mode="cloud", skills=["counting"])
    
    processor._last_results = {"count": 5}
    
    state = processor.state()
    assert "count" in state
    assert state["count"] == 5
    processor.close()


def test_state_empty_before_processing():
    """Test that state is empty before any processing."""
    processor = MoondreamProcessor(mode="cloud")
    state = processor.state()
    assert state == {}
    processor.close()


def test_summarize_detections():
    """Test detection summarization for LLM."""
    processor = MoondreamProcessor(mode="cloud")
    
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
    processor = MoondreamProcessor(mode="cloud")
    summary = processor._summarize_detections([])
    assert summary == "No objects detected"
    processor.close()

