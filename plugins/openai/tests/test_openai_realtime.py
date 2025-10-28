import asyncio
import pytest
from dotenv import load_dotenv

from vision_agents.plugins.openai import Realtime
from vision_agents.core.llm.events import (
    RealtimeAudioOutputEvent,
    RealtimeUserSpeechTranscriptionEvent,
    RealtimeAgentSpeechTranscriptionEvent,
)

# Load environment variables
load_dotenv()


class TestOpenAIRealtime:
    """Integration tests for OpenAI Realtime API"""

    @pytest.fixture
    async def realtime(self):
        """Create and manage Realtime connection lifecycle"""
        realtime = Realtime(
            model="gpt-realtime",
            voice="alloy",
        )
        try:
            yield realtime
        finally:
            await realtime.close()

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me?")

        # Wait for response
        await asyncio.sleep(3.0)
        assert len(events) > 0

    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()

        # Wait for connection to be fully established
        await asyncio.sleep(2.0)

        # Convert 16kHz audio to 48kHz for OpenAI realtime
        # OpenAI expects 48kHz PCM audio
        import numpy as np
        from scipy import signal
        from vision_agents.core.edge.types import PcmData

        # Resample from 16kHz to 48kHz
        samples_16k = mia_audio_16khz.samples
        num_samples_48k = int(len(samples_16k) * 48000 / 16000)
        samples_48k = signal.resample(samples_16k, num_samples_48k).astype(np.int16)

        # Create new PcmData with 48kHz
        audio_48khz = PcmData(samples=samples_48k, sample_rate=48000, format="s16")

        await realtime.simple_response(
            "Listen to the following audio and tell me what you hear"
        )
        await asyncio.sleep(5.0)

        # Send the resampled audio
        await realtime.simple_audio_response(audio_48khz)

        # Wait for response
        await asyncio.sleep(10.0)
        assert len(events) > 0

    @pytest.mark.integration
    async def test_video_sending_flow(self, realtime, bunny_video_track):
        """Test sending real video data and verify connection remains stable"""
        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Describe what you see in this video please")
        await asyncio.sleep(10.0)
        # Start video sender with low FPS to avoid overwhelming the connection
        await realtime._watch_video_track(bunny_video_track)

        # Let it run for a few seconds
        await asyncio.sleep(10.0)

        # Stop video sender
        await realtime._stop_watching_video_track()
        assert len(events) > 0

    async def test_user_speech_transcription_event(self, realtime):
        """Test that user speech transcription event is emitted when conversation.item.input_audio_transcription.completed is received"""
        user_transcripts = []

        # Subscribe using decorator pattern like agents.py does
        @realtime.events.subscribe
        async def on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
            user_transcripts.append(event)

        # Real OpenAI event payload for user speech transcription (actual payload from OpenAI)
        openai_event = {
            "content_index": 0,
            "event_id": "event_CSLB0tmlLaCQtfefQZW03",
            "item_id": "item_CSLAtg5dOSD0bC3yKdelc",
            "transcript": "OK, everybody. Do.",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 2, "type": "duration"},
        }

        # Simulate receiving the event from OpenAI
        await realtime._handle_openai_event(openai_event)

        # Wait for async event processing
        await asyncio.sleep(0.1)

        # Verify the event was emitted
        assert len(user_transcripts) == 1
        assert user_transcripts[0].text == "OK, everybody. Do."
        assert user_transcripts[0].original == openai_event

    async def test_agent_speech_transcription_event(self, realtime):
        """Test that agent speech transcription event is emitted when response.audio_transcript.done is received"""
        agent_transcripts = []

        # Subscribe using decorator pattern like agents.py does
        @realtime.events.subscribe
        async def on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent):
            agent_transcripts.append(event)

        # Real OpenAI event payload for agent speech transcription
        openai_event = {
            "type": "response.output_audio_transcript.done",
            "event_id": "event_789",
            "response_id": "resp_abc",
            "item_id": "item_def",
            "output_index": 0,
            "content_index": 0,
            "transcript": "I'm doing well, thank you for asking!",
        }

        # Simulate receiving the event from OpenAI
        await realtime._handle_openai_event(openai_event)

        # Wait a moment for async event handling
        await asyncio.sleep(0.1)

        # Verify the event was emitted
        assert len(agent_transcripts) == 1
        assert agent_transcripts[0].text == "I'm doing well, thank you for asking!"
        assert agent_transcripts[0].original == openai_event

    async def test_both_transcription_events(self, realtime):
        """Test that both user and agent transcription events are emitted correctly"""
        user_transcripts = []
        agent_transcripts = []

        # Subscribe using decorator pattern like agents.py does
        @realtime.events.subscribe
        async def on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
            user_transcripts.append(event)

        @realtime.events.subscribe
        async def on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent):
            agent_transcripts.append(event)

        # Real user speech event
        user_event = {
            "content_index": 0,
            "event_id": "event_user_123",
            "item_id": "item_user_456",
            "transcript": "Hello, how are you?",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 1, "type": "duration"},
        }

        # Real agent speech event
        agent_event = {
            "type": "response.output_audio_transcript.done",
            "event_id": "event_agent_789",
            "response_id": "resp_abc",
            "item_id": "item_agent_def",
            "output_index": 0,
            "content_index": 0,
            "transcript": "I'm doing great, thanks!",
        }

        # Simulate receiving both events
        await realtime._handle_openai_event(user_event)
        await realtime._handle_openai_event(agent_event)

        # Wait for async event processing
        await asyncio.sleep(0.1)

        # Verify both events were emitted
        assert len(user_transcripts) == 1
        assert user_transcripts[0].text == "Hello, how are you?"
        assert user_transcripts[0].original == user_event

        assert len(agent_transcripts) == 1
        assert agent_transcripts[0].text == "I'm doing great, thanks!"
        assert agent_transcripts[0].original == agent_event

    async def test_user_speech_transcription_tracks_participant(self, realtime):
        """Test that user speech transcription correctly tracks participant/user_id"""
        user_transcripts = []

        # Subscribe to events
        @realtime.events.subscribe
        async def on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
            user_transcripts.append(event)

        # Create a mock participant with a specific user_id
        from vision_agents.core.edge.types import Participant

        test_participant = Participant(original=None, user_id="test_user_123")

        # Simulate sending audio with participant info
        from vision_agents.core.edge.types import PcmData
        import numpy as np

        pcm = PcmData(
            samples=np.zeros(100, dtype=np.int16), sample_rate=48000, format="s16"
        )
        await realtime.simple_audio_response(pcm, test_participant)

        # Simulate OpenAI creating the conversation item (this is when we map item_id -> participant)
        item_created_event = {
            "type": "conversation.item.created",
            "event_id": "event_created_123",
            "item": {
                "id": "item_test_456",
                "type": "message",
                "role": "user",
                "content": [],
            },
        }
        await realtime._handle_openai_event(item_created_event)

        # Now simulate receiving the transcription event from OpenAI
        openai_event = {
            "content_index": 0,
            "event_id": "event_test_123",
            "item_id": "item_test_456",
            "transcript": "Test transcription with user ID",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 1, "type": "duration"},
        }

        await realtime._handle_openai_event(openai_event)
        await asyncio.sleep(0.1)

        # Verify the event was emitted with correct participant info
        assert len(user_transcripts) == 1
        assert user_transcripts[0].text == "Test transcription with user ID"

        # Verify the participant/user_id is correctly tracked
        assert user_transcripts[0].user_metadata is not None
        assert user_transcripts[0].user_metadata.user_id == "test_user_123"

        # Verify the user_id() helper method works
        assert user_transcripts[0].user_id() == "test_user_123"

    async def test_multi_user_participant_tracking(self, realtime):
        """Test that participant tracking works correctly when multiple users speak in succession"""
        user_transcripts = []

        @realtime.events.subscribe
        async def on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
            user_transcripts.append(event)

        from vision_agents.core.edge.types import Participant
        from vision_agents.core.edge.types import PcmData
        import numpy as np

        # User A sends audio
        participant_a = Participant(original=None, user_id="user_a")
        pcm_a = PcmData(
            samples=np.zeros(100, dtype=np.int16), sample_rate=48000, format="s16"
        )
        await realtime.simple_audio_response(pcm_a, participant_a)

        # OpenAI creates conversation item for User A
        item_created_a = {
            "type": "conversation.item.created",
            "event_id": "event_created_a",
            "item": {
                "id": "item_a_123",
                "type": "message",
                "role": "user",
                "content": [],
            },
        }
        await realtime._handle_openai_event(item_created_a)

        # User B sends audio (before A's transcription arrives)
        participant_b = Participant(original=None, user_id="user_b")
        pcm_b = PcmData(
            samples=np.zeros(100, dtype=np.int16), sample_rate=48000, format="s16"
        )
        await realtime.simple_audio_response(pcm_b, participant_b)

        # OpenAI creates conversation item for User B
        item_created_b = {
            "type": "conversation.item.created",
            "event_id": "event_created_b",
            "item": {
                "id": "item_b_456",
                "type": "message",
                "role": "user",
                "content": [],
            },
        }
        await realtime._handle_openai_event(item_created_b)

        # Now transcriptions arrive (A's transcription arrives AFTER B started speaking)
        transcription_a = {
            "content_index": 0,
            "event_id": "event_trans_a",
            "item_id": "item_a_123",  # References User A's item
            "transcript": "Hello from User A",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 1, "type": "duration"},
        }
        await realtime._handle_openai_event(transcription_a)

        transcription_b = {
            "content_index": 0,
            "event_id": "event_trans_b",
            "item_id": "item_b_456",  # References User B's item
            "transcript": "Hello from User B",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 1, "type": "duration"},
        }
        await realtime._handle_openai_event(transcription_b)

        await asyncio.sleep(0.1)

        # Verify both transcriptions are attributed to the correct users
        assert len(user_transcripts) == 2

        # User A's transcription should be attributed to User A (not B, despite B speaking more recently)
        assert user_transcripts[0].text == "Hello from User A"
        assert user_transcripts[0].user_metadata is not None
        assert user_transcripts[0].user_metadata.user_id == "user_a"

        # User B's transcription should be attributed to User B
        assert user_transcripts[1].text == "Hello from User B"
        assert user_transcripts[1].user_metadata is not None
        assert user_transcripts[1].user_metadata.user_id == "user_b"
