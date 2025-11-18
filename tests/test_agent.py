"""
Test suite for Agent core functionality.

Tests cover:
- wait_for_participant method
"""

import asyncio
from unittest.mock import Mock
import pytest

from getstream.video.rtc.participants import ParticipantsState
from vision_agents.core.agents.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.core.edge.sfu_events import Participant
from vision_agents.core.llm.llm import LLM
from vision_agents.core.stt.stt import STT


class MockLLM(LLM):
    """Mock LLM for testing"""

    async def simple_response(self, text: str, processors=None, participant=None):
        """Mock simple_response"""
        return Mock(text="mock response", original={})

    def _attach_agent(self, agent):
        """Mock attach agent"""
        pass


class MockSTT(STT):
    """Mock STT for testing"""

    async def process_audio(self, pcm, participant):
        """Mock process_audio"""
        pass

    def set_output_format(self, sample_rate, channels):
        """Mock set_output_format"""
        pass


class MockEdge:
    """Mock edge transport for testing"""

    def __init__(self):
        from vision_agents.core.events.manager import EventManager

        self.events = EventManager()
        self.client = Mock()

    async def create_user(self, user):
        """Mock create user"""
        pass

    def create_audio_track(self, framerate=48000, stereo=True):
        """Mock creating audio track"""
        return Mock(id="audio_track_1")


class TestAgentWaitForParticipant:
    """Test suite for Agent wait_for_participant logic"""

    def create_mock_agent(self, llm=None):
        """Helper to create a mock agent with minimal setup"""
        if llm is None:
            llm = MockLLM()

        edge = MockEdge()
        agent_user = User(id="test-agent", name="Test Agent")

        # Create agent with minimal config (need STT for validation)
        agent = Agent(
            edge=edge,
            llm=llm,
            agent_user=agent_user,
            instructions="Test instructions",
            stt=MockSTT(),
        )

        # Set up call and participants state
        agent.call = Mock(id="test-call")

        # Create a mock ParticipantsState with the needed behavior
        mock_participants = Mock(spec=ParticipantsState)
        mock_participants._participants = []
        mock_participants._callbacks = []

        def mock_map(callback):
            """Mock the map method to store callback and call it with current participants"""
            # Store callback for later updates
            mock_participants._callbacks.append(callback)
            # Call immediately with current state
            callback(mock_participants._participants)

            subscription = Mock()
            subscription.unsubscribe = Mock(
                side_effect=lambda: mock_participants._callbacks.remove(callback)
            )
            return subscription

        def trigger_update():
            """Helper to trigger all callbacks with current participants"""
            for cb in mock_participants._callbacks:
                cb(mock_participants._participants)

        mock_participants.map = mock_map
        mock_participants.trigger_update = trigger_update
        agent.participants = mock_participants

        return agent

    @pytest.mark.asyncio
    async def test_wait_for_participant_already_present(self):
        """Test that wait_for_participant returns immediately if participant already in call"""
        agent = self.create_mock_agent()

        # Add a non-agent participant to the call
        participant = Participant(user_id="user-1", session_id="session-1")
        agent.participants._participants.append(participant)

        # This should return immediately without waiting
        await asyncio.wait_for(agent.wait_for_participant(), timeout=1.0)

        # Test passes if we didn't timeout

    @pytest.mark.asyncio
    async def test_wait_for_participant_agent_doesnt_count(self):
        """Test that the agent itself in the call doesn't satisfy wait_for_participant"""
        agent = self.create_mock_agent()

        # Add only the agent to the call
        agent_participant = Participant(
            user_id=agent.agent_user.id, session_id="agent-session"
        )
        agent.participants._participants.append(agent_participant)

        # This should timeout since only agent is present
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.wait_for_participant(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_wait_for_participant_event_triggered(self):
        """Test that wait_for_participant completes when a participant joins"""
        agent = self.create_mock_agent()

        # No participants present initially (participants list is empty by default)

        # Create a task to wait for participant
        wait_task = asyncio.create_task(agent.wait_for_participant())

        # Give it a moment to set up the event handler
        await asyncio.sleep(0.1)

        # Task should be waiting
        assert not wait_task.done()

        # Add a participant to simulate someone joining
        new_participant = Participant(user_id="user-1", session_id="session-1")
        agent.participants._participants.append(new_participant)

        # Trigger the participants update to notify subscribers
        agent.participants.trigger_update()

        # Give it a moment to process
        await asyncio.sleep(0.05)

        # Wait task should complete now
        await asyncio.wait_for(wait_task, timeout=1.0)
