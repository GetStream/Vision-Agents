"""
Test suite for Agent core functionality.

Tests cover:
- wait_for_participant method
"""

import asyncio
from unittest.mock import Mock
import pytest

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
            log_level=None,
        )
        
        # Set up call and participants state
        agent.call = Mock(id="test-call")
        agent.participants = {}
        
        return agent
    
    @pytest.mark.asyncio
    async def test_wait_for_participant_already_present(self):
        """Test that wait_for_participant returns immediately if participant already in call"""
        agent = self.create_mock_agent()
        
        # Add a non-agent participant to the call
        participant = Participant(
            user_id="user-1",
            session_id="session-1"
        )
        agent.participants["session-1"] = participant
        
        # This should return immediately without waiting
        await asyncio.wait_for(agent.wait_for_participant(), timeout=1.0)
        
        # Test passes if we didn't timeout
        
    @pytest.mark.asyncio
    async def test_wait_for_participant_agent_doesnt_count(self):
        """Test that the agent itself in the call doesn't satisfy wait_for_participant"""
        agent = self.create_mock_agent()
        
        # Add only the agent to the call
        agent_participant = Participant(
            user_id=agent.agent_user.id,
            session_id="agent-session"
        )
        agent.participants["agent-session"] = agent_participant
        
        # This should timeout since only agent is present
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.wait_for_participant(), timeout=0.5)
    
    @pytest.mark.asyncio
    async def test_wait_for_participant_event_triggered(self):
        """Test that wait_for_participant completes when ParticipantJoinedEvent is triggered"""
        from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
        from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2
        from vision_agents.core.edge.sfu_events import ParticipantJoinedEvent
        
        agent = self.create_mock_agent()
        
        # Register the ParticipantJoinedEvent so it can be sent
        agent.edge.events.register(ParticipantJoinedEvent)
        
        # No participants present initially
        agent.participants = {}
        
        # Create a task to wait for participant
        wait_task = asyncio.create_task(agent.wait_for_participant())
        
        # Give it a moment to set up the event handler
        await asyncio.sleep(0.1)
        
        # Task should be waiting
        assert not wait_task.done()
        
        # Create a proper protobuf ParticipantJoined event
        participant_proto = models_pb2.Participant(
            user_id="user-1",
            session_id="session-1"
        )
        proto_event = events_pb2.ParticipantJoined(
            call_cid="test-call",
            participant=participant_proto
        )
        
        # Send the raw protobuf message (it gets auto-wrapped by the event manager)
        agent.edge.events.send(proto_event)
        
        # Wait for the event to be processed
        await agent.edge.events.wait()
        
        # Wait task should complete now
        await asyncio.wait_for(wait_task, timeout=1.0)

