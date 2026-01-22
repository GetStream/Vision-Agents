"""Tests for localrtc plugin."""

import pytest
from unittest.mock import patch

from vision_agents.plugins.localrtc import Edge, LocalRoom


def test_imports():
    """Test that basic imports work."""
    assert Edge is not None
    assert LocalRoom is not None


def test_local_room_creation():
    """Test LocalRoom creation."""
    room = LocalRoom(room_id="test-room")
    assert room.id == "test-room"
    assert room.type == "default"

    # Test with custom room_type
    room_custom = LocalRoom(room_id="test-room-2", room_type="local")
    assert room_custom.id == "test-room-2"
    assert room_custom.type == "local"


@pytest.mark.asyncio
async def test_local_room_leave():
    """Test LocalRoom leave method."""
    room = LocalRoom(room_id="test-room")
    await room.leave()


@pytest.mark.asyncio
async def test_local_edge_creation():
    """Test LocalEdge creation."""
    edge = Edge()
    assert edge is not None


def test_list_devices_static_method():
    """Test Edge.list_devices() static method returns correct structure."""
    mock_audio_inputs = [
        {"name": "Microphone 1", "index": 0},
        {"name": "Microphone 2", "index": 1},
    ]
    mock_audio_outputs = [
        {"name": "Speaker 1", "index": 0},
        {"name": "Speaker 2", "index": 1},
    ]
    mock_video_inputs = [
        {"name": "Camera 1", "index": 0},
    ]

    with patch("vision_agents.plugins.localrtc.edge.list_audio_inputs") as mock_ai, \
         patch("vision_agents.plugins.localrtc.edge.list_audio_outputs") as mock_ao, \
         patch("vision_agents.plugins.localrtc.edge.list_video_inputs") as mock_vi:

        mock_ai.return_value = mock_audio_inputs
        mock_ao.return_value = mock_audio_outputs
        mock_vi.return_value = mock_video_inputs

        devices = Edge.list_devices()

        assert "audio_inputs" in devices
        assert "audio_outputs" in devices
        assert "video_inputs" in devices

        assert devices["audio_inputs"] == mock_audio_inputs
        assert devices["audio_outputs"] == mock_audio_outputs
        assert devices["video_inputs"] == mock_video_inputs


def test_list_devices_empty_devices():
    """Test Edge.list_devices() with no devices available."""
    with patch("vision_agents.plugins.localrtc.edge.list_audio_inputs") as mock_ai, \
         patch("vision_agents.plugins.localrtc.edge.list_audio_outputs") as mock_ao, \
         patch("vision_agents.plugins.localrtc.edge.list_video_inputs") as mock_vi:

        mock_ai.return_value = []
        mock_ao.return_value = []
        mock_vi.return_value = []

        devices = Edge.list_devices()

        assert devices["audio_inputs"] == []
        assert devices["audio_outputs"] == []
        assert devices["video_inputs"] == []


def test_list_devices_is_static():
    """Test that list_devices can be called without an instance."""
    # Should be callable without creating an Edge instance
    with patch("vision_agents.plugins.localrtc.edge.list_audio_inputs") as mock_ai, \
         patch("vision_agents.plugins.localrtc.edge.list_audio_outputs") as mock_ao, \
         patch("vision_agents.plugins.localrtc.edge.list_video_inputs") as mock_vi:

        mock_ai.return_value = []
        mock_ao.return_value = []
        mock_vi.return_value = []

        # Call without creating an instance
        devices = Edge.list_devices()

        assert isinstance(devices, dict)
        assert len(devices) == 3
