"""Integration tests for Agent with Local RTC Edge.

This module tests the integration between the Agent and Local RTC Edge transport,
verifying that agents can be created, started, stopped, and that events flow correctly
when using local device access.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from vision_agents.core import Agent, User
from vision_agents.core.llm.llm import AudioLLM, LLMResponseEvent
from vision_agents.core.types import PcmData
from vision_agents.plugins.localrtc import Edge, LocalRoom


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
def mock_devices():
    """Mock device list for testing without hardware."""
    return {
        "audio_inputs": [
            {"name": "Mock Microphone", "index": 0},
        ],
        "audio_outputs": [
            {"name": "Mock Speaker", "index": 0},
        ],
        "video_inputs": [
            {"name": "Mock Camera", "index": 0},
        ],
    }


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice library to avoid hardware dependencies."""
    mock_devices = [
        {"name": "Mock Microphone", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Mock Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        mock_sd.InputStream = MagicMock()
        mock_sd.OutputStream = MagicMock()
        yield mock_sd


@pytest.fixture
def mock_opencv():
    """Mock OpenCV library to avoid hardware dependencies."""
    with patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.VideoCapture.return_value = mock_cap
        yield mock_cv2


class TestAgentLocalRTCIntegration:
    """Integration tests for Agent with Local RTC."""

    @pytest.mark.asyncio
    async def test_agent_creation_with_localrtc_edge(
        self, mock_sounddevice, mock_opencv
    ):
        """Test that Agent can be instantiated with localrtc.Edge."""
        # Create Local RTC Edge
        edge = Edge(
            audio_device="default",
            video_device=0,
            speaker_device="default",
            sample_rate=16000,
            channels=1,
        )

        # Create Agent with Local RTC Edge
        agent = Agent(
            edge=edge,
            llm=DummyAudioLLM(),
            agent_user=User(name="Test Agent", id="agent-test"),
            instructions="Test agent for integration testing",
        )

        # Verify agent was created successfully
        assert agent is not None
        assert agent.edge is edge
        assert isinstance(agent.llm, DummyAudioLLM)
        assert agent.agent_user.name == "Test Agent"

        # Clean up
        await agent.close()

    @pytest.mark.asyncio
    async def test_agent_start_and_stop(self, mock_sounddevice, mock_opencv):
        """Test that agent can start and stop correctly."""
        # Create Local RTC Edge
        edge = Edge(
            audio_device="default",
            video_device=0,
            speaker_device="default",
        )

        # Create Agent
        agent = Agent(
            edge=edge,
            llm=DummyAudioLLM(),
            agent_user=User(name="Test Agent", id="agent-test"),
        )

        # Create a local room
        room = LocalRoom(room_id="test-room-123", room_type="test")

        # Verify room properties
        assert room.id == "test-room-123"
        assert room.type == "test"

        # Test that agent can join and leave
        try:
            # Mock the edge methods to avoid actual device access
            with patch.object(edge, "create_user", new_callable=AsyncMock) as mock_create_user, \
                 patch.object(edge, "join", new_callable=AsyncMock) as mock_join, \
                 patch.object(edge, "publish_tracks", new_callable=AsyncMock), \
                 patch.object(edge, "create_conversation", new_callable=AsyncMock):

                mock_join.return_value = room

                # Start agent
                async with agent.join(room):
                    # Verify edge methods were called
                    mock_create_user.assert_called_once()
                    mock_join.assert_called_once()

                    # Agent is running in context
                    assert agent is not None

                # After exiting context, agent should have cleaned up
                # (Room leave is called in the context manager)
        finally:
            # Clean up
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_with_mock_devices(self, mock_devices):
        """Test agent creation with mocked device access."""
        with patch("vision_agents.plugins.localrtc.edge.list_audio_inputs") as mock_ai, \
             patch("vision_agents.plugins.localrtc.edge.list_audio_outputs") as mock_ao, \
             patch("vision_agents.plugins.localrtc.edge.list_video_inputs") as mock_vi, \
             patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd, \
             patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:

            # Setup device mocks
            mock_ai.return_value = mock_devices["audio_inputs"]
            mock_ao.return_value = mock_devices["audio_outputs"]
            mock_vi.return_value = mock_devices["video_inputs"]

            # Setup sounddevice mock
            mock_sd.query_devices.return_value = [
                {"name": "Mock Microphone", "max_input_channels": 2, "max_output_channels": 0},
                {"name": "Mock Speaker", "max_input_channels": 0, "max_output_channels": 2},
            ]

            # Setup OpenCV mock
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            # Create edge and verify devices are available
            devices = Edge.list_devices()
            assert len(devices["audio_inputs"]) > 0
            assert len(devices["audio_outputs"]) > 0
            assert len(devices["video_inputs"]) > 0

            # Create Agent with mocked devices
            edge = Edge(audio_device="default", video_device=0)
            agent = Agent(
                edge=edge,
                llm=DummyAudioLLM(),
                agent_user=User(name="Test Agent", id="agent-test"),
            )

            assert agent is not None
            await agent.close()

    @pytest.mark.asyncio
    async def test_events_flow_correctly(self, mock_sounddevice, mock_opencv):
        """Test that events flow correctly through the agent."""
        # Create Local RTC Edge
        edge = Edge(audio_device="default", video_device=0)

        # Create Agent
        agent = Agent(
            edge=edge,
            llm=DummyAudioLLM(),
            agent_user=User(name="Test Agent", id="agent-test"),
        )

        # Track events
        events_received = []

        def on_event(event):
            events_received.append(event)

        # Subscribe to edge events
        edge.on("track_subscribed", on_event)

        try:
            # Create room
            room = LocalRoom(room_id="test-events-room")

            # Mock edge methods
            with patch.object(edge, "create_user", new_callable=AsyncMock), \
                 patch.object(edge, "join", new_callable=AsyncMock) as mock_join, \
                 patch.object(edge, "publish_tracks", new_callable=AsyncMock), \
                 patch.object(edge, "create_conversation", new_callable=AsyncMock):

                mock_join.return_value = room

                # Join room
                async with agent.join(room):
                    # Simulate an event
                    edge.emit("track_subscribed", {"track_id": "test-track"})

                    # Give event time to propagate
                    await asyncio.sleep(0.1)

                # Verify event was received
                assert len(events_received) > 0
                assert events_received[0]["track_id"] == "test-track"
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_with_no_hardware(self):
        """Test agent works without actual hardware by mocking all device access."""
        with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd, \
             patch("vision_agents.plugins.localrtc.tracks.cv2") as mock_cv2:

            # Mock all device queries to return empty/mock devices
            mock_sd.query_devices.return_value = [
                {"name": "Virtual Mic", "max_input_channels": 1, "max_output_channels": 0},
                {"name": "Virtual Speaker", "max_input_channels": 0, "max_output_channels": 2},
            ]

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, MagicMock())
            mock_cv2.VideoCapture.return_value = mock_cap

            # Create edge without real hardware
            edge = Edge(audio_device="default", video_device=0)

            # Create agent
            agent = Agent(
                edge=edge,
                llm=DummyAudioLLM(),
                agent_user=User(name="Virtual Agent", id="agent-virtual"),
            )

            # Verify agent was created
            assert agent is not None
            assert agent.edge is edge

            # Clean up
            await agent.close()

    @pytest.mark.asyncio
    async def test_room_leave_idempotency(self):
        """Test that room leave can be called multiple times without error."""
        room = LocalRoom(room_id="test-idempotent-room")

        # Leave multiple times should not raise error
        await room.leave()
        await room.leave()
        await room.leave()

        # Room properties should still be accessible
        assert room.id == "test-idempotent-room"
        assert room.type == "default"

    @pytest.mark.asyncio
    async def test_agent_simple_response(self, mock_sounddevice, mock_opencv):
        """Test that agent can generate simple responses."""
        edge = Edge(audio_device="default", video_device=0)
        llm = DummyAudioLLM()
        agent = Agent(
            edge=edge,
            llm=llm,
            agent_user=User(name="Test Agent", id="agent-test"),
        )

        try:
            # Test simple response - Agent.simple_response() returns None
            # but we can verify it was called without error
            await agent.simple_response("Hello, how are you?")

            # Verify the agent is still functional
            assert agent is not None
            assert agent.llm is llm
        finally:
            await agent.close()
