import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from getstream.models import ChannelMemberRequest
from getstream.video.rtc import ConnectionManager
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from vision_agents.plugins.getstream.stream_edge_transport import (
    StreamConnection,
    StreamEdge,
)


@pytest.fixture
async def stream_edge(monkeypatch):
    monkeypatch.setenv("STREAM_API_KEY", "test-key")
    monkeypatch.setenv("STREAM_API_SECRET", "test-secret")
    return StreamEdge()


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


class TestStreamEdge:
    async def test_create_call_raises_before_authenticate(
        self, stream_edge: StreamEdge
    ):
        with pytest.raises(RuntimeError, match="not authenticated"):
            await stream_edge.create_call(call_id="call-1")

    async def test_open_demo_update_uses_channel_member_request(
        self, stream_edge: StreamEdge
    ):
        """When user not in members, update must use ChannelMemberRequest (v3 API)."""
        stream_edge._agent_user_id = "agent-user"

        mock_channel = AsyncMock()
        # User NOT in members list -> triggers the update(add_members=...) branch
        mock_channel.get_or_create.return_value = Mock(data=Mock(members=[]))

        mock_call = Mock()
        mock_call.id = "test-call-id"
        mock_call.client.stream = stream_edge.client

        with (
            patch.object(stream_edge.client, "create_user", new_callable=AsyncMock),
            patch.object(stream_edge.client.chat, "channel", return_value=mock_channel),
            patch.object(stream_edge.client, "create_token", return_value="fake-token"),
            patch("webbrowser.open"),
        ):
            await stream_edge.open_demo(mock_call)

        # Verify channel.update was called with ChannelMemberRequest
        mock_channel.update.assert_called_once()
        call_kwargs = mock_channel.update.call_args.kwargs
        add_members = call_kwargs["add_members"]
        assert len(add_members) == 1
        assert isinstance(add_members[0], ChannelMemberRequest)
        assert add_members[0].user_id == "user-demo-agent"
