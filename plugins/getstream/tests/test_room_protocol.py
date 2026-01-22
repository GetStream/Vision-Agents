"""Unit tests for GetStreamRoomAdapter Room protocol compliance."""

import pytest
from unittest.mock import AsyncMock, Mock

from vision_agents.core.protocols import Room
from vision_agents.plugins.getstream.adapters import GetStreamRoomAdapter


class TestGetStreamRoomAdapterProtocolCompliance:
    """Test suite verifying GetStreamRoomAdapter satisfies Room protocol."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock GetStream Call object
        self.mock_call = Mock()
        self.mock_call.id = "getstream-call-123"
        self.mock_call.call_type = "default"
        self.mock_call.end = AsyncMock()

    def test_getstream_adapter_has_id_property(self):
        """Test that GetStreamRoomAdapter implements id property."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        assert hasattr(adapter, "id")
        assert isinstance(adapter.id, str)

    def test_getstream_adapter_has_type_property(self):
        """Test that GetStreamRoomAdapter implements type property."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        assert hasattr(adapter, "type")
        assert isinstance(adapter.type, str)

    def test_getstream_adapter_has_leave_method(self):
        """Test that GetStreamRoomAdapter implements leave method."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        assert hasattr(adapter, "leave")
        assert callable(adapter.leave)

    def test_getstream_adapter_id_returns_call_id(self):
        """Test that GetStreamRoomAdapter.id returns the call identifier."""
        call_id = "my-getstream-call-789"
        self.mock_call.id = call_id
        adapter = GetStreamRoomAdapter(self.mock_call)
        assert adapter.id == call_id

    def test_getstream_adapter_type_returns_call_type(self):
        """Test that GetStreamRoomAdapter.type returns the call type."""
        call_type = "rtc"
        self.mock_call.call_type = call_type
        adapter = GetStreamRoomAdapter(self.mock_call)
        assert adapter.type == call_type

    @pytest.mark.asyncio
    async def test_getstream_adapter_leave_is_async(self):
        """Test that GetStreamRoomAdapter.leave is an async method."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        # Should be callable and return a coroutine
        result = adapter.leave()
        assert hasattr(result, "__await__")
        await result

    @pytest.mark.asyncio
    async def test_getstream_adapter_leave_calls_end(self):
        """Test that GetStreamRoomAdapter.leave delegates to Call.end()."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        await adapter.leave()
        # Verify that the underlying call's end() method was called
        self.mock_call.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_getstream_adapter_leave_executes_successfully(self):
        """Test that GetStreamRoomAdapter.leave executes without errors."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        # Should complete without raising exceptions
        await adapter.leave()

    @pytest.mark.asyncio
    async def test_getstream_adapter_leave_is_idempotent(self):
        """Test that GetStreamRoomAdapter.leave can be called multiple times safely."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        # Multiple calls should work
        await adapter.leave()
        await adapter.leave()
        await adapter.leave()
        # end() should be called each time
        assert self.mock_call.end.call_count == 3

    def test_getstream_adapter_isinstance_check(self):
        """Test that GetStreamRoomAdapter can be checked against Room protocol."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        # Test structural compliance
        assert hasattr(adapter, "id")
        assert hasattr(adapter, "type")
        assert hasattr(adapter, "leave")

    def test_getstream_adapter_satisfies_protocol_interface(self):
        """Test that GetStreamRoomAdapter implements all required protocol methods."""
        adapter = GetStreamRoomAdapter(self.mock_call)

        # Check all protocol requirements are met
        assert hasattr(adapter, "id") and isinstance(
            getattr(type(adapter), "id", None), property
        )
        assert hasattr(adapter, "type") and isinstance(
            getattr(type(adapter), "type", None), property
        )
        assert hasattr(adapter, "leave") and callable(adapter.leave)

    @pytest.mark.asyncio
    async def test_getstream_adapter_full_lifecycle(self):
        """Test complete GetStreamRoomAdapter lifecycle following protocol."""
        # Create adapter
        call_id = "lifecycle-test-call"
        call_type = "test-rtc"
        self.mock_call.id = call_id
        self.mock_call.call_type = call_type

        adapter = GetStreamRoomAdapter(self.mock_call)

        # Verify properties
        assert adapter.id == call_id
        assert adapter.type == call_type

        # Leave room
        await adapter.leave()
        self.mock_call.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_getstream_adapter_can_be_used_as_protocol(self):
        """Test that GetStreamRoomAdapter can be used where Room protocol is expected."""
        mock_call = Mock()
        mock_call.id = "test-call"
        mock_call.call_type = "default"
        mock_call.end = AsyncMock()

        room: Room = GetStreamRoomAdapter(mock_call)

        # Should be usable as a Room protocol object
        assert isinstance(room.id, str)
        assert isinstance(room.type, str)
        await room.leave()

    def test_getstream_adapter_id_is_property(self):
        """Test that GetStreamRoomAdapter.id is a property, not a method."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        # Should be accessible without calling
        adapter_id = adapter.id
        assert isinstance(adapter_id, str)

    def test_getstream_adapter_type_is_property(self):
        """Test that GetStreamRoomAdapter.type is a property, not a method."""
        adapter = GetStreamRoomAdapter(self.mock_call)
        # Should be accessible without calling
        adapter_type = adapter.type
        assert isinstance(adapter_type, str)

    def test_getstream_adapter_wraps_call_correctly(self):
        """Test that GetStreamRoomAdapter correctly wraps the Call object."""
        call_id = "wrapped-call-123"
        call_type = "wrapped-type"
        self.mock_call.id = call_id
        self.mock_call.call_type = call_type

        adapter = GetStreamRoomAdapter(self.mock_call)

        # Adapter should expose Call properties through protocol interface
        assert adapter.id == call_id
        assert adapter.type == call_type

    @pytest.mark.asyncio
    async def test_getstream_adapter_delegates_to_call(self):
        """Test that GetStreamRoomAdapter delegates operations to the wrapped Call."""
        adapter = GetStreamRoomAdapter(self.mock_call)

        # Call leave() on adapter
        await adapter.leave()

        # Verify delegation to Call.end()
        self.mock_call.end.assert_called_once()

    def test_getstream_adapter_properties_reflect_call_state(self):
        """Test that GetStreamRoomAdapter properties reflect the wrapped Call's state."""
        # Change call properties
        self.mock_call.id = "updated-id"
        self.mock_call.call_type = "updated-type"

        adapter = GetStreamRoomAdapter(self.mock_call)

        # Adapter should reflect updated values
        assert adapter.id == "updated-id"
        assert adapter.type == "updated-type"
