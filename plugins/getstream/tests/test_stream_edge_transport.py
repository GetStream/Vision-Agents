import asyncio
import time
from unittest.mock import Mock
from uuid import uuid4

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
        assert conn.idle_for() > 0

        # One participant (itself), still idle
        connection_manager.participants_state._add_participant(
            Participant(user_id=str(connection_manager.user_id))
        )
        time.sleep(0.01)
        assert conn.idle_for() > 0

        # A participant joined, not idle anymore
        another_participant = Participant(user_id="another-user-id")
        connection_manager.participants_state._add_participant(another_participant)
        time.sleep(0.01)
        assert not conn.idle_for()

        # A participant left, idle again
        connection_manager.participants_state._remove_participant(another_participant)
        time.sleep(0.01)
        assert conn.idle_for() > 0

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
