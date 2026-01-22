"""Tests for BaseGStreamerTrack abstract class."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock GStreamer before importing tracks
mock_gst = MagicMock()
mock_gst.State.PLAYING = 3
mock_gst.State.NULL = 1
mock_gst.StateChangeReturn.FAILURE = 0
mock_gst.StateChangeReturn.SUCCESS = 1
mock_gst.SECOND = 1000000000

# Patch modules before importing
sys.modules["gi"] = MagicMock()
sys.modules["gi.repository"] = MagicMock()

# Import needs to happen after module patching, so ignore E402
from vision_agents.plugins.localrtc.tracks import BaseGStreamerTrack  # noqa: E402


# Import with patched Gst module
@pytest.fixture(autouse=True)
def mock_gstreamer():
    """Mock GStreamer for all tests."""
    with patch("vision_agents.plugins.localrtc.tracks.Gst", mock_gst):
        with patch("vision_agents.plugins.localrtc.tracks.GST_AVAILABLE", True):
            yield


class ConcreteGStreamerTrack(BaseGStreamerTrack):
    """Concrete implementation of BaseGStreamerTrack for testing."""

    def _create_pipeline(self) -> None:
        """Create a mock pipeline."""
        self._pipeline = MagicMock()
        self._pipeline.set_state = MagicMock(return_value=mock_gst.StateChangeReturn.SUCCESS)


def test_base_gstreamer_track_init():
    """Test BaseGStreamerTrack initialization."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")
    assert track.pipeline_str == "fakesrc ! fakesink"
    assert track._pipeline is None
    assert track._running is False
    assert track._pipeline_thread is None


def test_base_gstreamer_track_init_no_gstreamer():
    """Test BaseGStreamerTrack initialization when GStreamer is not available."""
    with patch("vision_agents.plugins.localrtc.tracks.GST_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="GStreamer is not available"):
            ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")


def test_base_gstreamer_track_start_pipeline_success():
    """Test _start_pipeline successfully starts the pipeline."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")
    track._create_pipeline()

    track._start_pipeline()

    track._pipeline.set_state.assert_called_once_with(mock_gst.State.PLAYING)


def test_base_gstreamer_track_start_pipeline_no_pipeline():
    """Test _start_pipeline raises error when pipeline is not created."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    with pytest.raises(RuntimeError, match="Pipeline not created"):
        track._start_pipeline()


def test_base_gstreamer_track_start_pipeline_failure():
    """Test _start_pipeline handles pipeline start failure."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")
    track._create_pipeline()
    track._pipeline.set_state = MagicMock(return_value=mock_gst.StateChangeReturn.FAILURE)

    with pytest.raises(RuntimeError, match="Failed to start GStreamer pipeline"):
        track._start_pipeline()


def test_base_gstreamer_track_stop_pipeline():
    """Test _stop_pipeline stops the pipeline and cleans up."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")
    track._create_pipeline()

    # Keep a reference to the pipeline before it's set to None
    pipeline_ref = track._pipeline

    track._stop_pipeline()

    pipeline_ref.set_state.assert_called_with(mock_gst.State.NULL)
    assert track._pipeline is None


def test_base_gstreamer_track_stop_pipeline_no_pipeline():
    """Test _stop_pipeline safely handles when pipeline is None."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    # Should not raise error
    track._stop_pipeline()
    assert track._pipeline is None


def test_base_gstreamer_track_stop_pipeline_with_error():
    """Test _stop_pipeline handles errors gracefully."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")
    track._create_pipeline()
    track._pipeline.set_state = MagicMock(side_effect=Exception("Stop error"))

    # Should not raise error, just log warning
    track._stop_pipeline()
    assert track._pipeline is None


def test_base_gstreamer_track_wait_for_thread():
    """Test _wait_for_thread waits for pipeline thread to finish."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    # Create a mock thread
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    track._pipeline_thread = mock_thread

    track._wait_for_thread(timeout=1.0)

    mock_thread.join.assert_called_once_with(timeout=1.0)
    assert track._pipeline_thread is None


def test_base_gstreamer_track_wait_for_thread_no_thread():
    """Test _wait_for_thread safely handles when thread is None."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    # Should not raise error
    track._wait_for_thread()


def test_base_gstreamer_track_wait_for_thread_not_alive():
    """Test _wait_for_thread safely handles when thread is not alive."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    # Create a mock thread that's not alive
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False
    track._pipeline_thread = mock_thread

    track._wait_for_thread()

    # Should not call join on a dead thread
    mock_thread.join.assert_not_called()


def test_base_gstreamer_track_handle_pipeline_error():
    """Test _handle_pipeline_error formats errors consistently."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    error = Exception("Test error")
    result = track._handle_pipeline_error(error, "test operation")

    assert isinstance(result, RuntimeError)
    assert "test operation" in str(result)
    assert "Test error" in str(result)


def test_base_gstreamer_track_abstract_create_pipeline():
    """Test that _create_pipeline is abstract and must be implemented."""
    # Create a class that doesn't implement _create_pipeline
    class IncompleteTrack(BaseGStreamerTrack):
        pass

    # Should not be able to instantiate without implementing abstract method
    with pytest.raises(TypeError):
        IncompleteTrack(pipeline="fakesrc ! fakesink")


def test_base_gstreamer_track_pipeline_lifecycle():
    """Test complete pipeline lifecycle: create -> start -> stop."""
    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    # Create pipeline
    track._create_pipeline()
    assert track._pipeline is not None

    # Start pipeline
    track._start_pipeline()
    track._pipeline.set_state.assert_called_with(mock_gst.State.PLAYING)

    # Stop pipeline
    track._stop_pipeline()
    assert track._pipeline is None


def test_base_gstreamer_track_thread_lifecycle():
    """Test thread lifecycle with wait_for_thread."""
    import threading
    import time

    track = ConcreteGStreamerTrack(pipeline="fakesrc ! fakesink")

    # Create a real thread for testing
    def worker():
        time.sleep(0.1)

    track._pipeline_thread = threading.Thread(target=worker)
    track._pipeline_thread.start()

    # Wait for thread
    track._wait_for_thread(timeout=1.0)

    # Thread should be cleaned up
    assert track._pipeline_thread is None
