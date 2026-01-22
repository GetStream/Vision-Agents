"""Tests for localrtc plugin."""

import pytest

from vision_agents.plugins.localrtc import Edge, LocalRoom


def test_imports():
    """Test that basic imports work."""
    assert Edge is not None
    assert LocalRoom is not None


def test_local_room_creation():
    """Test LocalRoom creation."""
    room = LocalRoom(room_id="test-room")
    assert room.id == "test-room"
    assert room.type == "local"


@pytest.mark.asyncio
async def test_local_room_leave():
    """Test LocalRoom leave method."""
    room = LocalRoom(room_id="test-room")
    await room.leave()


def test_local_edge_creation():
    """Test LocalEdge creation."""
    edge = Edge()
    assert edge is not None
