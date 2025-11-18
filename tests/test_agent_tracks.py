"""
Test suite for Agent track handling logic.

Tests cover:
- Track priority (screenshare > regular video)
- Track forwarding to processors
- Processed track forwarding to LLM
- No tracks scenario
"""

import asyncio
from unittest.mock import Mock
import pytest

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from vision_agents.core.agents.agents import Agent
from vision_agents.core.edge.types import User, Participant
from vision_agents.core.llm.llm import LLM, VideoLLM
from vision_agents.core.processors.base_processor import Processor, ProcessorType


class MockVideoTrack:
    """Mock video track for testing"""

    def __init__(self, track_id: str):
        self.id = track_id
        self.kind = "video"

    def stop(self):
        pass


class MockVideoProcessor(Processor):
    """Mock video processor that tracks calls"""

    def __init__(self, name: str = "test_processor"):
        self.name = name
        self.process_video_calls = []
        self.processor_type = ProcessorType.VIDEO

    async def process_video(self, track, user_id, shared_forwarder=None):
        """Track that this processor received a video track"""
        self.process_video_calls.append(
            {"track": track, "user_id": user_id, "shared_forwarder": shared_forwarder}
        )


class MockVideoLLM(VideoLLM):
    """Mock VideoLLM that tracks video track assignments"""

    def __init__(self):
        super().__init__()
        self.watch_video_track_calls = []

    async def watch_video_track(self, track, shared_forwarder=None):
        """Track that the LLM received a video track"""
        self.watch_video_track_calls.append(
            {"track": track, "shared_forwarder": shared_forwarder}
        )

    async def simple_response(self, text: str, processors=None, participant=None):
        """Mock simple_response"""
        return Mock(text="mock response", original={})

    def _attach_agent(self, agent):
        """Mock attach agent"""
        pass


class MockLLM(LLM):
    """Mock LLM for non-video tests"""

    async def simple_response(self, text: str, processors=None, participant=None):
        """Mock simple_response"""
        return Mock(text="mock response", original={})

    def _attach_agent(self, agent):
        """Mock attach agent"""
        pass


class MockEdge:
    """Mock edge transport for testing"""

    def __init__(self):
        from vision_agents.core.events.manager import EventManager

        self.events = EventManager()
        self.add_track_subscriber_calls = []
        self.client = Mock()

    async def create_user(self, user):
        """Mock create user"""
        pass

    def add_track_subscriber(self, track_id):
        """Mock adding a track subscriber"""
        self.add_track_subscriber_calls.append(track_id)
        return MockVideoTrack(track_id)

    def create_audio_track(self, framerate=48000, stereo=True):
        """Mock creating audio track"""
        return Mock(id="audio_track_1")


class TestAgentTrackHandling:
    """Test suite for Agent track handling logic"""

    def create_mock_agent(self, llm=None, processors=None, tts=None, stt=None):
        """Helper to create a mock agent with minimal setup"""
        if llm is None:
            llm = MockLLM()

        if processors is None:
            processors = []

        edge = MockEdge()
        agent_user = User(id="test-agent", name="Test Agent")

        # Create agent with minimal config
        agent = Agent(
            edge=edge,
            llm=llm,
            agent_user=agent_user,
            instructions="Test instructions",
            processors=processors,
            tts=tts,
            stt=stt,
        )

        return agent

    @pytest.mark.asyncio
    async def test_no_video_tracks_nothing_shared(self):
        """Test that with no video tracks, nothing is forwarded to processors or LLM"""
        video_processor = MockVideoProcessor("processor1")
        video_llm = MockVideoLLM()

        agent = self.create_mock_agent(llm=video_llm, processors=[video_processor])

        # Verify no tracks were added
        assert len(agent._active_video_tracks) == 0

        # Verify processor was not called
        assert len(video_processor.process_video_calls) == 0

        # Verify LLM was not called
        assert len(video_llm.watch_video_track_calls) == 0

    @pytest.mark.asyncio
    async def test_regular_video_track_is_forwarded(self):
        """Test that a regular video track is forwarded to processor and LLM"""
        video_processor = MockVideoProcessor("processor1")
        video_llm = MockVideoLLM()

        agent = self.create_mock_agent(llm=video_llm, processors=[video_processor])

        # Simulate adding a video track
        participant = Participant(original=None, user_id="user-1")

        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        # Give async tasks time to complete
        await asyncio.sleep(0.1)

        # Verify track was added
        assert "video_track_1" in agent._active_video_tracks
        track_info = agent._active_video_tracks["video_track_1"]
        assert track_info.type == TrackType.TRACK_TYPE_VIDEO
        assert track_info.priority == 0  # Regular video has priority 0

        # Verify processor received the track
        assert len(video_processor.process_video_calls) == 1
        call = video_processor.process_video_calls[0]
        assert call["user_id"] == "user-1"
        assert call["shared_forwarder"] is not None

        # Verify LLM received the track
        assert len(video_llm.watch_video_track_calls) == 1
        llm_call = video_llm.watch_video_track_calls[0]
        assert llm_call["shared_forwarder"] is not None

    @pytest.mark.asyncio
    async def test_screenshare_takes_priority_over_video(self):
        """Test that screenshare track takes priority over regular video track"""
        video_processor = MockVideoProcessor("processor1")
        video_llm = MockVideoLLM()

        agent = self.create_mock_agent(llm=video_llm, processors=[video_processor])

        participant = Participant(original=None, user_id="user-1")

        # Add regular video track first
        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify regular video was processed
        assert len(video_processor.process_video_calls) == 1
        assert len(video_llm.watch_video_track_calls) == 1

        # Clear the calls
        video_processor.process_video_calls.clear()
        video_llm.watch_video_track_calls.clear()

        # Now add screenshare track
        await agent._on_track_added(
            track_id="screenshare_track_1",
            track_type=TrackType.TRACK_TYPE_SCREEN_SHARE,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify screenshare track was added with higher priority
        assert "screenshare_track_1" in agent._active_video_tracks
        screenshare_info = agent._active_video_tracks["screenshare_track_1"]
        assert screenshare_info.priority == 1  # Screenshare has priority 1

        video_info = agent._active_video_tracks["video_track_1"]
        assert video_info.priority == 0  # Regular video has priority 0

        # Verify processor received the screenshare track (higher priority)
        assert len(video_processor.process_video_calls) == 1
        call = video_processor.process_video_calls[0]
        # The processor should receive the highest priority NON-PROCESSED track (screenshare)
        # Note: TrackType.TRACK_TYPE_SCREEN_SHARE has value 3
        assert "screenshare_track_1" in call["shared_forwarder"].name

        # Verify LLM received the screenshare track (highest priority overall)
        assert len(video_llm.watch_video_track_calls) == 1
        llm_call = video_llm.watch_video_track_calls[0]
        # LLM should receive the highest priority track (screenshare)
        assert "screenshare_track_1" in llm_call["shared_forwarder"].name

    @pytest.mark.asyncio
    async def test_track_removed_updates_active_tracks(self):
        """Test that removing a track updates the active tracks"""
        video_processor = MockVideoProcessor("processor1")
        video_llm = MockVideoLLM()

        agent = self.create_mock_agent(llm=video_llm, processors=[video_processor])

        participant = Participant(original=None, user_id="user-1")

        # Add two video tracks
        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        await agent._on_track_added(
            track_id="screenshare_track_1",
            track_type=TrackType.TRACK_TYPE_SCREEN_SHARE,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        assert len(agent._active_video_tracks) == 2

        # Clear calls
        video_processor.process_video_calls.clear()
        video_llm.watch_video_track_calls.clear()

        # Remove screenshare track
        await agent._on_track_removed(
            track_id="screenshare_track_1",
            track_type=TrackType.TRACK_TYPE_SCREEN_SHARE,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify screenshare was removed
        assert "screenshare_track_1" not in agent._active_video_tracks
        assert "video_track_1" in agent._active_video_tracks

        # Verify processor now receives the regular video track (fallback)
        assert len(video_processor.process_video_calls) == 1
        call = video_processor.process_video_calls[0]
        assert "video_track_1" in call["shared_forwarder"].name

        # Verify LLM now receives the regular video track
        assert len(video_llm.watch_video_track_calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_processors_all_receive_tracks(self):
        """Test that all video processors receive the track"""
        processor1 = MockVideoProcessor("processor1")
        processor2 = MockVideoProcessor("processor2")
        video_llm = MockVideoLLM()

        agent = self.create_mock_agent(
            llm=video_llm, processors=[processor1, processor2]
        )

        participant = Participant(original=None, user_id="user-1")

        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify both processors received the track
        assert len(processor1.process_video_calls) == 1
        assert len(processor2.process_video_calls) == 1

        # Verify LLM received the track
        assert len(video_llm.watch_video_track_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_receives_highest_priority_track(self):
        """Test that LLM receives the highest priority track (including processed tracks)"""
        video_processor = MockVideoProcessor("processor1")
        video_llm = MockVideoLLM()

        agent = self.create_mock_agent(llm=video_llm, processors=[video_processor])

        participant = Participant(original=None, user_id="user-1")

        # Add regular video track
        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify the track info
        track_info = agent._active_video_tracks["video_track_1"]
        assert track_info.priority == 0
        assert track_info.processor == ""  # Non-processed track

        # Verify LLM received the track
        assert len(video_llm.watch_video_track_calls) == 1
        llm_call = video_llm.watch_video_track_calls[0]

        # The LLM should receive the track via shared_forwarder
        assert llm_call["shared_forwarder"] is not None
        assert llm_call["track"] is not None

    @pytest.mark.asyncio
    async def test_processors_optional(self):
        """Test that processors are optional and LLM still receives tracks"""
        video_llm = MockVideoLLM()

        # Create agent with no processors but with TTS to satisfy validation
        # (VideoLLM alone doesn't count as a processing capability)
        from vision_agents.core.events.manager import EventManager

        mock_tts = Mock()
        mock_tts.set_output_format = Mock()
        mock_tts.events = EventManager()  # TTS needs an events manager

        agent = self.create_mock_agent(llm=video_llm, processors=[], tts=mock_tts)

        participant = Participant(original=None, user_id="user-1")

        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify track was added
        assert "video_track_1" in agent._active_video_tracks

        # Verify LLM still received the track directly
        assert len(video_llm.watch_video_track_calls) == 1

    @pytest.mark.asyncio
    async def test_non_video_llm_does_not_receive_tracks(self):
        """Test that non-VideoLLM implementations don't receive tracks"""
        regular_llm = MockLLM()
        video_processor = MockVideoProcessor("processor1")

        agent = self.create_mock_agent(llm=regular_llm, processors=[video_processor])

        participant = Participant(original=None, user_id="user-1")

        await agent._on_track_added(
            track_id="video_track_1",
            track_type=TrackType.TRACK_TYPE_VIDEO,
            participant=participant,
        )

        await asyncio.sleep(0.1)

        # Verify track was added and processor received it
        assert "video_track_1" in agent._active_video_tracks
        assert len(video_processor.process_video_calls) == 1

        # Verify regular LLM doesn't have watch_video_track method called
        assert not hasattr(regular_llm, "watch_video_track_calls")
