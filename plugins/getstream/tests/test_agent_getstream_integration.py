"""Integration tests for Agent with GetStream Edge.

This module tests backward compatibility after refactoring, ensuring that:
- GetStream edge still works with Agent
- Existing GetStream workflows function correctly
- Type adapters convert correctly between GetStream and core types
- Events use core types but GetStream integration remains functional

These tests verify that the refactoring to use core types hasn't broken the
GetStream edge transport integration.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    Participant,
    TrackType as StreamTrackType,
)
from vision_agents.core import Agent, User
from vision_agents.core.llm.llm import AudioLLM, LLMResponseEvent
from vision_agents.core.types import PcmData, TrackType
from vision_agents.plugins.getstream import Edge, adapt_pcm_data, adapt_track_type
from vision_agents.plugins.getstream.adapters import GetStreamRoomAdapter


class DummyAudioLLM(AudioLLM):
    """Dummy AudioLLM implementation for testing.

    This is a realtime audio LLM that doesn't require separate STT/TTS services.
    """

    async def simple_response(self, *_, **__) -> LLMResponseEvent[Any]:
        """Return a simple mock response."""
        return LLMResponseEvent(text="Test response", original=None)

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        """Handle audio input and return audio response."""
        pass


@pytest.fixture
def mock_getstream_client():
    """Mock GetStream client for testing without API calls."""
    mock_client = Mock()
    mock_client.video = Mock()
    return mock_client


@pytest.fixture
def mock_call():
    """Mock GetStream Call object."""
    mock = Mock()
    mock.id = "test-call-123"
    mock.call_type = "default"
    mock.end = AsyncMock()
    return mock


class TestAgentGetStreamIntegration:
    """Integration tests for Agent with GetStream Edge."""

    @pytest.mark.asyncio
    async def test_agent_creation_with_getstream_edge(self, mock_getstream_client):
        """Test that Agent can be instantiated with GetStream Edge."""
        # Create GetStream Edge
        with patch(
            "vision_agents.plugins.getstream.stream_edge_transport.AsyncStream"
        ) as mock_stream:
            mock_stream.return_value = mock_getstream_client

            edge = Edge(
                api_key="test-api-key",
                api_secret="test-api-secret",
            )

            # Create Agent with GetStream Edge
            agent = Agent(
                edge=edge,
                llm=DummyAudioLLM(),
                agent_user=User(name="Test Agent", id="agent-test"),
                instructions="Test agent for GetStream integration testing",
            )

            # Verify agent was created successfully
            assert agent is not None
            assert agent.edge is edge
            assert isinstance(agent.llm, DummyAudioLLM)
            assert agent.agent_user.name == "Test Agent"

            # Clean up
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_join_getstream_call(self, mock_getstream_client, mock_call):
        """Test that agent can join and leave GetStream calls."""
        with patch(
            "vision_agents.plugins.getstream.stream_edge_transport.AsyncStream"
        ) as mock_stream:
            mock_stream.return_value = mock_getstream_client

            edge = Edge(
                api_key="test-api-key",
                api_secret="test-api-secret",
            )

            agent = Agent(
                edge=edge,
                llm=DummyAudioLLM(),
                agent_user=User(name="Test Agent", id="agent-test"),
            )

            try:
                # Mock the edge methods
                with patch.object(
                    edge, "create_user", new_callable=AsyncMock
                ) as mock_create_user, patch.object(
                    edge, "join", new_callable=AsyncMock
                ) as mock_join, patch.object(
                    edge, "publish_tracks", new_callable=AsyncMock
                ), patch.object(
                    edge, "create_conversation", new_callable=AsyncMock
                ):
                    # Create a mock StreamConnection
                    mock_connection = MagicMock()
                    mock_connection.id = "test-call-123"
                    mock_connection.type = "default"
                    mock_connection.wait_for_participant = AsyncMock()
                    mock_connection.close = AsyncMock()
                    mock_connection.leave = AsyncMock()

                    mock_join.return_value = mock_connection

                    # Start agent and join call
                    async with agent.join(mock_connection):
                        # Verify edge methods were called
                        mock_create_user.assert_called_once()
                        mock_join.assert_called_once()

                        # Verify room properties
                        assert mock_connection.id == "test-call-123"
                        assert mock_connection.type == "default"

                    # After exiting context, connection.close() should have been called
                    mock_connection.close.assert_called()
            finally:
                await agent.close()

    @pytest.mark.asyncio
    async def test_getstream_room_adapter_with_agent(self, mock_call):
        """Test that GetStreamRoomAdapter works correctly with Agent."""
        # Create room adapter
        room = GetStreamRoomAdapter(mock_call)

        # Verify Room protocol compliance
        assert room.id == "test-call-123"
        assert room.type == "default"
        assert hasattr(room, "leave")
        assert callable(room.leave)

        # Verify leave delegates to call.end()
        await room.leave()
        mock_call.end.assert_called_once()

    def test_type_adapter_pcm_data_conversion(self):
        """Test that PCM data adapters convert correctly between GetStream and core types."""
        # Create a mock StreamPcmData with expected interface
        stream_pcm = Mock()
        stream_pcm.data = b"\x00\x01\x02\x03\x04\x05"
        stream_pcm.sample_rate = 48000
        stream_pcm.channels = 2
        stream_pcm.bit_depth = 16

        # Convert to core PCM data
        core_pcm = adapt_pcm_data(stream_pcm)

        # Verify conversion
        assert isinstance(core_pcm, PcmData)
        assert core_pcm.data == stream_pcm.data
        assert core_pcm.sample_rate == stream_pcm.sample_rate
        assert core_pcm.channels == stream_pcm.channels
        assert core_pcm.bit_depth == stream_pcm.bit_depth

    def test_type_adapter_track_type_audio_conversion(self):
        """Test that audio track types convert correctly."""
        # Test TRACK_TYPE_AUDIO
        core_type = adapt_track_type(StreamTrackType.TRACK_TYPE_AUDIO)
        assert core_type == TrackType.AUDIO

        # Test TRACK_TYPE_SCREEN_SHARE_AUDIO
        core_type = adapt_track_type(StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO)
        assert core_type == TrackType.AUDIO

    def test_type_adapter_track_type_video_conversion(self):
        """Test that video track types convert correctly."""
        # Test TRACK_TYPE_VIDEO
        core_type = adapt_track_type(StreamTrackType.TRACK_TYPE_VIDEO)
        assert core_type == TrackType.VIDEO

    def test_type_adapter_track_type_screenshare_conversion(self):
        """Test that screenshare track types convert correctly."""
        # Test TRACK_TYPE_SCREEN_SHARE
        core_type = adapt_track_type(StreamTrackType.TRACK_TYPE_SCREEN_SHARE)
        assert core_type == TrackType.SCREENSHARE

    def test_type_adapter_track_type_unknown_defaults_to_video(self):
        """Test that unknown track types default to VIDEO."""
        # Test unknown track type
        unknown_type = 999
        core_type = adapt_track_type(unknown_type)
        assert core_type == TrackType.VIDEO

    @pytest.mark.asyncio
    async def test_events_use_core_types(self, mock_getstream_client):
        """Test that events use core types but GetStream still works."""
        with patch(
            "vision_agents.plugins.getstream.stream_edge_transport.AsyncStream"
        ) as mock_stream:
            mock_stream.return_value = mock_getstream_client

            edge = Edge(
                api_key="test-api-key",
                api_secret="test-api-secret",
            )

            # Track events
            events_received = []

            def on_track_added(event):
                """Track added event handler."""
                events_received.append(event)

            # Subscribe to track events
            edge.on("track_added", on_track_added)

            # Simulate a track added event with core types
            test_event = {
                "track_type": TrackType.AUDIO,
                "track_id": "test-track-123",
                "user_id": "test-user",
            }
            edge.emit("track_added", test_event)

            # Give event time to propagate
            await asyncio.sleep(0.01)

            # Verify event was received with core types
            assert len(events_received) == 1
            assert events_received[0]["track_type"] == TrackType.AUDIO
            assert isinstance(events_received[0]["track_type"], TrackType)

    @pytest.mark.asyncio
    async def test_getstream_workflows_still_function(
        self, mock_getstream_client, mock_call
    ):
        """Test that existing GetStream workflows still function after refactoring."""
        with patch(
            "vision_agents.plugins.getstream.stream_edge_transport.AsyncStream"
        ) as mock_stream:
            mock_stream.return_value = mock_getstream_client

            # Create edge
            edge = Edge(
                api_key="test-api-key",
                api_secret="test-api-secret",
            )

            # Create agent
            agent = Agent(
                edge=edge,
                llm=DummyAudioLLM(),
                agent_user=User(name="Test Agent", id="agent-test"),
            )

            try:
                # Mock edge methods to simulate full workflow
                with patch.object(
                    edge, "create_user", new_callable=AsyncMock
                ) as mock_create_user, patch.object(
                    edge, "join", new_callable=AsyncMock
                ) as mock_join, patch.object(
                    edge, "publish_tracks", new_callable=AsyncMock
                ), patch.object(
                    edge, "create_conversation", new_callable=AsyncMock
                ):
                    # Create a mock StreamConnection
                    mock_connection = MagicMock()
                    mock_connection.id = "test-call-123"
                    mock_connection.type = "default"
                    mock_connection.wait_for_participant = AsyncMock()
                    mock_connection.close = AsyncMock()
                    mock_connection.leave = AsyncMock()

                    mock_join.return_value = mock_connection

                    # Workflow: create user -> join -> publish tracks -> conversation
                    async with agent.join(mock_connection):
                        # Verify workflow steps were executed
                        mock_create_user.assert_called_once()
                        mock_join.assert_called_once()

                    # Connection should have been closed
                    mock_connection.close.assert_called()

            finally:
                await agent.close()

    @pytest.mark.asyncio
    async def test_pcm_data_with_timestamp(self):
        """Test that PCM data adapter preserves timestamp if present."""
        # Create mock GetStream PCM data with timestamp
        stream_pcm = Mock()
        stream_pcm.data = b"\x00\x01"
        stream_pcm.sample_rate = 48000
        stream_pcm.channels = 1
        stream_pcm.bit_depth = 16
        stream_pcm.timestamp = 123.456

        # Convert to core PCM data
        core_pcm = adapt_pcm_data(stream_pcm)

        # Verify timestamp is preserved
        assert core_pcm.timestamp == 123.456

    @pytest.mark.asyncio
    async def test_pcm_data_without_timestamp(self):
        """Test that PCM data adapter handles missing timestamp."""
        # Create mock GetStream PCM data without timestamp
        stream_pcm = Mock(spec=["data", "sample_rate", "channels", "bit_depth"])
        stream_pcm.data = b"\x00\x01"
        stream_pcm.sample_rate = 48000
        stream_pcm.channels = 1
        stream_pcm.bit_depth = 16

        # Convert to core PCM data
        core_pcm = adapt_pcm_data(stream_pcm)

        # Verify timestamp defaults to None
        assert core_pcm.timestamp is None

    @pytest.mark.asyncio
    async def test_room_adapter_idempotent_leave(self, mock_call):
        """Test that GetStreamRoomAdapter.leave() is idempotent."""
        room = GetStreamRoomAdapter(mock_call)

        # Call leave multiple times
        await room.leave()
        await room.leave()
        await room.leave()

        # Verify call.end() was called each time (idempotency is delegated to GetStream)
        assert mock_call.end.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_simple_response_with_getstream(self, mock_getstream_client):
        """Test that agent can generate simple responses with GetStream edge."""
        with patch(
            "vision_agents.plugins.getstream.stream_edge_transport.AsyncStream"
        ) as mock_stream:
            mock_stream.return_value = mock_getstream_client

            edge = Edge(
                api_key="test-api-key",
                api_secret="test-api-secret",
            )

            llm = DummyAudioLLM()
            agent = Agent(
                edge=edge,
                llm=llm,
                agent_user=User(name="Test Agent", id="agent-test"),
            )

            try:
                # Test simple response
                await agent.simple_response("Hello, how are you?")

                # Verify agent is still functional
                assert agent is not None
                assert agent.llm is llm
            finally:
                await agent.close()

    def test_all_stream_track_types_have_core_equivalents(self):
        """Test that all GetStream track types have valid core type mappings."""
        # Test all defined GetStream track types
        track_type_mappings = {
            StreamTrackType.TRACK_TYPE_AUDIO: TrackType.AUDIO,
            StreamTrackType.TRACK_TYPE_VIDEO: TrackType.VIDEO,
            StreamTrackType.TRACK_TYPE_SCREEN_SHARE: TrackType.SCREENSHARE,
            StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO: TrackType.AUDIO,
        }

        for stream_type, expected_core_type in track_type_mappings.items():
            core_type = adapt_track_type(stream_type)
            assert (
                core_type == expected_core_type
            ), f"Failed mapping {stream_type} to {expected_core_type}"

    @pytest.mark.asyncio
    async def test_edge_maintains_getstream_api_surface(self, mock_getstream_client):
        """Test that Edge maintains the expected GetStream API surface."""
        with patch(
            "vision_agents.plugins.getstream.stream_edge_transport.AsyncStream"
        ) as mock_stream:
            mock_stream.return_value = mock_getstream_client

            edge = Edge(
                api_key="test-api-key",
                api_secret="test-api-secret",
            )

            # Verify Edge has expected methods
            assert hasattr(edge, "create_user")
            assert hasattr(edge, "join")
            assert hasattr(edge, "publish_tracks")
            assert hasattr(edge, "create_conversation")
            assert hasattr(edge, "create_audio_track")
            assert hasattr(edge, "create_video_track")

            # Verify methods are callable
            assert callable(edge.create_user)
            assert callable(edge.join)
            assert callable(edge.publish_tracks)
            assert callable(edge.create_conversation)
