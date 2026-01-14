import asyncio
import time
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import aiortc.exceptions
import pytest
from getstream.video.rtc import ConnectionManager
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from vision_agents.plugins.getstream.stream_edge_transport import StreamConnection


@pytest.fixture
def connection_manager():
    return ConnectionManager(user_id=str(uuid4()), call=Mock())


class TestStreamConnection:
    def test_idle_for(self, connection_manager):
        # No participants, connection is idle
        conn = StreamConnection(connection=connection_manager)
        time.sleep(0.01)
        assert conn.idle_since() > 0

        # One participant (itself), still idle
        connection_manager.participants_state._add_participant(
            Participant(user_id=str(connection_manager.user_id))
        )
        time.sleep(0.01)
        assert conn.idle_since() > 0

        # A participant joined, not idle anymore
        another_participant = Participant(user_id="another-user-id")
        connection_manager.participants_state._add_participant(another_participant)
        time.sleep(0.01)
        assert not conn.idle_since()

        # A participant left, idle again
        connection_manager.participants_state._remove_participant(another_participant)
        time.sleep(0.01)
        assert conn.idle_since() > 0

    async def test_wait_for_participant_already_present(self, connection_manager):
        """Test that wait_for_participant returns immediately if participant already in call"""

        conn = StreamConnection(connection_manager)
        # Add a non-agent participant to the call
        participant = Participant(user_id="user-1", session_id="session-1")
        connection_manager.participants_state._add_participant(participant)

        # This should return immediately without waiting
        await asyncio.wait_for(conn.wait_for_participant(), timeout=1.0)

    async def test_wait_for_participant_agent_doesnt_count(self, connection_manager):
        """
        Test that the agent itself in the call doesn't satisfy wait_for_participant
        """
        conn = StreamConnection(connection_manager)
        # Add only the agent to the call
        agent_participant = Participant(
            user_id=connection_manager.user_id, session_id="session-1"
        )
        connection_manager.participants_state._add_participant(agent_participant)

        # This should timeout since only agent is present
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.wait_for_participant(timeout=2.0), timeout=0.5)

    async def test_wait_for_participant_event_triggered(self, connection_manager):
        """Test that wait_for_participant completes when a participant joins"""
        # No participants present initially (participants list is empty by default)
        conn = StreamConnection(connection_manager)

        # Create a task to wait for participant
        wait_task = asyncio.create_task(conn.wait_for_participant())

        # Give it a moment to set up the event handler
        await asyncio.sleep(0.1)

        # Task should be waiting
        assert not wait_task.done()

        # Add a participant to simulate someone joining
        participant = Participant(user_id="user-1", session_id="session-1")
        connection_manager.participants_state._add_participant(participant)

        # Give it a moment to process
        await asyncio.sleep(0.05)

        # Wait task should complete now
        await asyncio.wait_for(wait_task, timeout=1.0)

    async def test_close_handles_invalid_state_error(self, connection_manager):
        """Test that close() handles InvalidStateError when connection is already closed.

        This race condition occurs when the call ends while the agent is still joining.
        """
        conn = StreamConnection(connection_manager)
        # Mock leave() to raise InvalidStateError (simulating closed WebRTC transport)
        connection_manager.leave = AsyncMock(
            side_effect=aiortc.exceptions.InvalidStateError("RTCIceTransport is closed")
        )

        # Should not raise, just log and return gracefully
        await conn.close()

    async def test_close_handles_local_description_attribute_error(
        self, connection_manager
    ):
        """Test that close() handles AttributeError for localDescription.

        This race condition occurs when WebRTC resources are None due to the call
        ending during connection setup.
        """
        conn = StreamConnection(connection_manager)
        # Mock leave() to raise AttributeError (simulating None subscriber_pc)
        connection_manager.leave = AsyncMock(
            side_effect=AttributeError(
                "'NoneType' object has no attribute 'localDescription'"
            )
        )

        # Should not raise, just log and return gracefully
        await conn.close()

    async def test_close_reraises_unrelated_attribute_error(self, connection_manager):
        """Test that close() re-raises unrelated AttributeErrors."""
        conn = StreamConnection(connection_manager)
        connection_manager.leave = AsyncMock(
            side_effect=AttributeError("some_unrelated_attribute")
        )

        with pytest.raises(AttributeError, match="some_unrelated_attribute"):
            await conn.close()

    async def test_close_handles_timeout(self, connection_manager):
        """Test that close() handles timeout when leave() takes too long."""
        conn = StreamConnection(connection_manager)

        async def slow_leave():
            await asyncio.sleep(10)

        connection_manager.leave = slow_leave

        # Should not raise, just log warning and return
        await conn.close(timeout=0.1)


class TestRaceConditionResilience:
    """Tests that verify resilience to race conditions during join/close."""

    async def test_close_during_active_operations(self, connection_manager):
        """Simulate close() being called while operations are in flight.

        This mimics the real race condition where the call ends while the agent
        is still joining or performing other operations.
        """
        conn = StreamConnection(connection_manager)
        errors_to_test = [
            aiortc.exceptions.InvalidStateError("RTCIceTransport is closed"),
            aiortc.exceptions.InvalidStateError("RTCDtlsTransport is closed"),
            AttributeError("'NoneType' object has no attribute 'localDescription'"),
            AttributeError("'NoneType' object has no attribute 'sdp'"),
        ]

        for error in errors_to_test:
            connection_manager.leave = AsyncMock(side_effect=error)
            # Should not raise for any of these expected race condition errors
            await conn.close()

    async def test_rapid_close_calls(self, connection_manager):
        """Test that multiple rapid close() calls don't cause issues."""
        conn = StreamConnection(connection_manager)
        call_count = 0

        async def counting_leave():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)

        connection_manager.leave = counting_leave

        # Call close multiple times rapidly
        await asyncio.gather(
            conn.close(),
            conn.close(),
            conn.close(),
        )

        # Leave should be called (at least once, implementation may dedupe)
        assert call_count >= 1
