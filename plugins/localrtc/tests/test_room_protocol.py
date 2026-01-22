"""Unit tests for LocalRoom Room protocol compliance."""

import pytest

from vision_agents.core.protocols import Room
from vision_agents.plugins.localrtc.room import LocalRoom


class TestLocalRoomProtocolCompliance:
    """Test suite verifying LocalRoom satisfies Room protocol."""

    def test_local_room_has_id_property(self):
        """Test that LocalRoom implements id property."""
        room = LocalRoom(room_id="test-room-123")
        assert hasattr(room, "id")
        assert isinstance(room.id, str)

    def test_local_room_has_type_property(self):
        """Test that LocalRoom implements type property."""
        room = LocalRoom(room_id="test-room-123", room_type="local")
        assert hasattr(room, "type")
        assert isinstance(room.type, str)

    def test_local_room_has_leave_method(self):
        """Test that LocalRoom implements leave method."""
        room = LocalRoom(room_id="test-room-123")
        assert hasattr(room, "leave")
        assert callable(room.leave)

    def test_local_room_id_returns_correct_value(self):
        """Test that LocalRoom.id returns the room identifier."""
        room_id = "my-local-room-456"
        room = LocalRoom(room_id=room_id)
        assert room.id == room_id

    def test_local_room_type_returns_correct_value(self):
        """Test that LocalRoom.type returns the room type."""
        room_type = "custom-type"
        room = LocalRoom(room_id="test-room", room_type=room_type)
        assert room.type == room_type

    def test_local_room_type_default_value(self):
        """Test that LocalRoom.type has a default value."""
        room = LocalRoom(room_id="test-room")
        assert isinstance(room.type, str)
        assert room.type == "default"

    @pytest.mark.asyncio
    async def test_local_room_leave_is_async(self):
        """Test that LocalRoom.leave is an async method."""
        room = LocalRoom(room_id="test-room")
        # Should be callable and return a coroutine
        result = room.leave()
        assert hasattr(result, "__await__")
        await result

    @pytest.mark.asyncio
    async def test_local_room_leave_executes_successfully(self):
        """Test that LocalRoom.leave executes without errors."""
        room = LocalRoom(room_id="test-room")
        # Should complete without raising exceptions
        await room.leave()

    @pytest.mark.asyncio
    async def test_local_room_leave_is_idempotent(self):
        """Test that LocalRoom.leave can be called multiple times safely."""
        room = LocalRoom(room_id="test-room")
        # First call
        await room.leave()
        # Second call should not raise an error
        await room.leave()
        # Third call should also be safe
        await room.leave()

    def test_local_room_isinstance_check(self):
        """Test that LocalRoom can be checked against Room protocol using isinstance."""
        room = LocalRoom(room_id="test-room")
        # Note: isinstance checks with Protocol require runtime_checkable decorator
        # This test documents the structural typing behavior
        assert hasattr(room, "id")
        assert hasattr(room, "type")
        assert hasattr(room, "leave")

    def test_local_room_satisfies_protocol_interface(self):
        """Test that LocalRoom implements all required protocol methods."""
        room = LocalRoom(room_id="test-room")

        # Check all protocol requirements are met
        assert hasattr(room, "id") and isinstance(
            getattr(type(room), "id", None), property
        )
        assert hasattr(room, "type") and isinstance(
            getattr(type(room), "type", None), property
        )
        assert hasattr(room, "leave") and callable(room.leave)

    @pytest.mark.asyncio
    async def test_local_room_full_lifecycle(self):
        """Test complete LocalRoom lifecycle following protocol."""
        # Create room
        room_id = "lifecycle-test-room"
        room_type = "test"
        room = LocalRoom(room_id=room_id, room_type=room_type)

        # Verify properties
        assert room.id == room_id
        assert room.type == room_type

        # Leave room
        await room.leave()

    @pytest.mark.asyncio
    async def test_local_room_can_be_used_as_protocol(self):
        """Test that LocalRoom can be used where Room protocol is expected."""
        room: Room = LocalRoom(room_id="test-room", room_type="local")

        # Should be usable as a Room protocol object
        assert isinstance(room.id, str)
        assert isinstance(room.type, str)
        await room.leave()

    def test_local_room_id_is_property(self):
        """Test that LocalRoom.id is a property, not a method."""
        room = LocalRoom(room_id="test-room")
        # Should be accessible without calling
        room_id = room.id
        assert isinstance(room_id, str)

    def test_local_room_type_is_property(self):
        """Test that LocalRoom.type is a property, not a method."""
        room = LocalRoom(room_id="test-room")
        # Should be accessible without calling
        room_type = room.type
        assert isinstance(room_type, str)

    def test_local_room_properties_are_read_only_by_convention(self):
        """Test that LocalRoom properties follow read-only convention."""
        room = LocalRoom(room_id="original-id", room_type="original-type")

        # Properties should return consistent values
        assert room.id == "original-id"
        assert room.type == "original-type"

        # Check multiple accesses return same values
        assert room.id == room.id
        assert room.type == room.type
