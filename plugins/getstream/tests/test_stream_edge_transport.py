import asyncio
import time
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from getstream.models import QueryUsersPayload, UserRequest
from getstream.video.rtc import ConnectionManager
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from vision_agents.core.edge.types import User
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

    async def test_close_releases_client_resources(self, stream_edge: StreamEdge):
        stream_edge._real_connection = AsyncMock()
        real_connection = stream_edge._real_connection

        assert stream_edge.client.client.is_closed is False
        assert stream_edge._real_connection is not None

        await stream_edge.close()

        assert stream_edge.client.client.is_closed is True
        assert stream_edge._real_connection is None
        real_connection.leave.assert_called_once()

    @pytest.mark.integration
    async def test_authenticate_keeps_custom_fields_set_elsewhere(self) -> None:
        """Metadata is merged into the stored user instead of replacing it."""
        edge = StreamEdge()
        user_id = f"test-authenticate-{uuid4()}"
        try:
            await edge.client.upsert_users(
                UserRequest(
                    id=user_id, name="Stored name", custom={"set_elsewhere": "keep me"}
                )
            )

            await edge.authenticate(
                User(id=user_id, name="Agent", custom={"is_agent": True})
            )

            response = await edge.client.query_users(
                QueryUsersPayload(filter_conditions={"id": {"$eq": user_id}})
            )
            stored = response.data.users[0]
            assert stored.custom == {"set_elsewhere": "keep me", "is_agent": True}
            assert stored.name == "Agent"
        finally:
            await edge.close()

    @pytest.mark.integration
    async def test_authenticate_unsets_none_custom_fields(self) -> None:
        """None custom values remove stored fields without touching unrelated ones."""
        edge = StreamEdge()
        user_id = f"test-authenticate-unset-{uuid4()}"
        try:
            await edge.client.upsert_users(
                UserRequest(
                    id=user_id,
                    name="Stored name",
                    custom={
                        "set_elsewhere": "keep me",
                        "stt": {"provider": "deepgram"},
                    },
                )
            )

            await edge.authenticate(
                User(
                    id=user_id,
                    name="Agent",
                    custom={"is_agent": True, "stt": None},
                )
            )

            response = await edge.client.query_users(
                QueryUsersPayload(filter_conditions={"id": {"$eq": user_id}})
            )
            stored = response.data.users[0]
            assert stored.custom == {"set_elsewhere": "keep me", "is_agent": True}
            assert "stt" not in stored.custom
        finally:
            await edge.close()

    @pytest.mark.integration
    async def test_authenticate_omits_none_custom_on_create(self) -> None:
        """404 create path omits null-valued custom fields instead of storing them."""
        edge = StreamEdge()
        user_id = f"test-authenticate-create-none-{uuid4()}"
        try:
            await edge.authenticate(
                User(
                    id=user_id,
                    name="Agent",
                    custom={"is_agent": True, "stt": None},
                )
            )

            response = await edge.client.query_users(
                QueryUsersPayload(filter_conditions={"id": {"$eq": user_id}})
            )
            stored = response.data.users[0]
            assert stored.custom == {"is_agent": True}
            assert "stt" not in stored.custom
        finally:
            await edge.close()

    @pytest.mark.integration
    async def test_create_users_creates_users_that_do_not_exist_yet(self) -> None:
        edge = StreamEdge()
        user_id = f"test-create-users-{uuid4()}"
        try:
            created = await edge.create_users(
                [User(id=user_id, name="New user", custom={"is_agent": False})]
            )

            assert [u.id for u in created] == [user_id]
            assert created[0].name == "New user"
            assert created[0].custom == {"is_agent": False}
        finally:
            await edge.close()
