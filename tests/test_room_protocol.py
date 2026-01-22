"""Unit tests for Room protocol compliance.

This module tests that implementations satisfy the Room protocol defined in
vision_agents.core.protocols. The Room protocol defines the interface that
edge transport implementations must provide.

Note: This test file contains general protocol tests and tests for implementations
that can be tested without additional dependencies. Plugin-specific implementations
like LocalRoom and GetStreamRoomAdapter are tested in their respective plugin
test directories.
"""

import pytest
from unittest.mock import AsyncMock

from vision_agents.core.protocols import Room


class TestRoomProtocol:
    """Test suite for Room protocol definition."""

    def test_room_is_protocol(self):
        """Test that Room is a Protocol class."""
        # Protocol classes have special attributes
        assert hasattr(Room, "__protocol_attrs__")

    def test_room_protocol_has_id_property(self):
        """Test that Room protocol defines id property."""
        # Check that the protocol has 'id' as an attribute
        assert hasattr(Room, "id")

    def test_room_protocol_has_type_property(self):
        """Test that Room protocol defines type property."""
        # Check that the protocol has 'type' as an attribute
        assert hasattr(Room, "type")

    def test_room_protocol_has_leave_method(self):
        """Test that Room protocol defines leave method."""
        # Check that the protocol has 'leave' as an attribute
        assert hasattr(Room, "leave")


class TestRoomProtocolWithMockImplementation:
    """Test suite for Room protocol using a mock implementation."""

    class MockRoom:
        """Mock implementation of Room protocol for testing."""

        def __init__(self, room_id: str, room_type: str = "mock"):
            self._id = room_id
            self._type = room_type

        @property
        def id(self) -> str:
            return self._id

        @property
        def type(self) -> str:
            return self._type

        async def leave(self) -> None:
            pass

    def test_mock_room_has_id_property(self):
        """Test that mock Room implementation has id property."""
        room = self.MockRoom(room_id="test-123")
        assert hasattr(room, "id")
        assert isinstance(room.id, str)
        assert room.id == "test-123"

    def test_mock_room_has_type_property(self):
        """Test that mock Room implementation has type property."""
        room = self.MockRoom(room_id="test-123", room_type="custom")
        assert hasattr(room, "type")
        assert isinstance(room.type, str)
        assert room.type == "custom"

    def test_mock_room_has_leave_method(self):
        """Test that mock Room implementation has leave method."""
        room = self.MockRoom(room_id="test-123")
        assert hasattr(room, "leave")
        assert callable(room.leave)

    @pytest.mark.asyncio
    async def test_mock_room_leave_is_async(self):
        """Test that mock Room.leave is an async method."""
        room = self.MockRoom(room_id="test-123")
        result = room.leave()
        assert hasattr(result, "__await__")
        await result

    @pytest.mark.asyncio
    async def test_mock_room_can_be_used_as_protocol(self):
        """Test that mock Room can be used where Room protocol is expected."""
        room: Room = self.MockRoom(room_id="test-123", room_type="protocol-test")

        # Should be usable as a Room protocol object
        assert isinstance(room.id, str)
        assert isinstance(room.type, str)
        assert room.id == "test-123"
        assert room.type == "protocol-test"
        await room.leave()

    def test_mock_room_satisfies_protocol_interface(self):
        """Test that mock Room implements all required protocol methods."""
        room = self.MockRoom(room_id="test-123")

        # Check all protocol requirements are met
        assert hasattr(room, "id") and isinstance(
            getattr(type(room), "id", None), property
        )
        assert hasattr(room, "type") and isinstance(
            getattr(type(room), "type", None), property
        )
        assert hasattr(room, "leave") and callable(room.leave)


class TestRoomProtocolStructuralTyping:
    """Test suite for Room protocol structural typing behavior."""

    def test_any_class_with_protocol_interface_can_be_used(self):
        """Test that any class implementing the protocol interface can be used as Room."""

        class CustomRoom:
            @property
            def id(self) -> str:
                return "custom-id"

            @property
            def type(self) -> str:
                return "custom-type"

            async def leave(self) -> None:
                pass

        room = CustomRoom()
        # Should have all protocol members
        assert hasattr(room, "id")
        assert hasattr(room, "type")
        assert hasattr(room, "leave")

    @pytest.mark.asyncio
    async def test_protocol_allows_duck_typing(self):
        """Test that Room protocol allows duck typing."""

        class DuckRoom:
            """A class that looks like a Room but doesn't explicitly inherit from it."""

            def __init__(self):
                self._room_id = "duck-123"
                self._room_type = "duck"

            @property
            def id(self) -> str:
                return self._room_id

            @property
            def type(self) -> str:
                return self._room_type

            async def leave(self) -> None:
                """Duck-typed leave method."""
                pass

        # Can be used as a Room protocol object
        room: Room = DuckRoom()
        assert room.id == "duck-123"
        assert room.type == "duck"
        await room.leave()
