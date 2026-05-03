import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import AudioFormat, PcmData
from scipy import signal
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.realtime import (
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeUserTranscript,
)
from vision_agents.plugins.openai import Realtime

load_dotenv()


class TestOpenAIRealtime:
    """Unit tests for OpenAI Realtime event handling."""

    @pytest.fixture
    async def realtime(self):
        # AsyncMock client avoids constructing AsyncOpenAI (which trips on SOCKS-proxy
        # envs missing the optional httpx[socks] extra) and lets unit tests drive
        # _handle_openai_event without a network roundtrip.
        rt = Realtime(model="gpt-realtime", voice="alloy", client=AsyncMock())
        rt.set_instructions("be friendly")
        try:
            yield rt
        finally:
            await rt.close()

    async def test_user_speech_transcription_event(self, realtime):
        """conversation.item.input_audio_transcription.completed → RealtimeUserTranscript"""
        realtime._current_participant = Participant(original=None, user_id="u", id="u")

        # Real OpenAI event payload for user speech transcription
        openai_event = {
            "content_index": 0,
            "event_id": "event_CSLB0tmlLaCQtfefQZW03",
            "item_id": "item_CSLAtg5dOSD0bC3yKdelc",
            "transcript": "OK, everybody. Do.",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 2, "type": "duration"},
        }

        await realtime._handle_openai_event(openai_event)

        user_transcripts = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeUserTranscript)
        ]
        assert len(user_transcripts) == 1
        assert user_transcripts[0].text == "OK, everybody. Do."

    async def test_agent_speech_transcription_event(self, realtime):
        """response.output_audio_transcript.done → RealtimeAgentTranscript"""
        openai_event = {
            "type": "response.output_audio_transcript.done",
            "event_id": "event_789",
            "response_id": "resp_abc",
            "item_id": "item_def",
            "output_index": 0,
            "content_index": 0,
            "transcript": "I'm doing well, thank you for asking!",
        }

        await realtime._handle_openai_event(openai_event)

        agent_transcripts = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAgentTranscript)
        ]
        assert len(agent_transcripts) == 1
        assert agent_transcripts[0].text == "I'm doing well, thank you for asking!"

    async def test_both_transcription_events(self, realtime):
        """Both user and agent transcripts are emitted on the output stream."""
        realtime._current_participant = Participant(original=None, user_id="u", id="u")

        user_event = {
            "content_index": 0,
            "event_id": "event_user_123",
            "item_id": "item_user_456",
            "transcript": "Hello, how are you?",
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"seconds": 1, "type": "duration"},
        }
        agent_event = {
            "type": "response.output_audio_transcript.done",
            "event_id": "event_agent_789",
            "response_id": "resp_abc",
            "item_id": "item_agent_def",
            "output_index": 0,
            "content_index": 0,
            "transcript": "I'm doing great, thanks!",
        }

        await realtime._handle_openai_event(user_event)
        await realtime._handle_openai_event(agent_event)

        items = realtime.output.peek()
        user_transcripts = [i for i in items if isinstance(i, RealtimeUserTranscript)]
        agent_transcripts = [i for i in items if isinstance(i, RealtimeAgentTranscript)]
        assert len(user_transcripts) == 1
        assert user_transcripts[0].text == "Hello, how are you?"
        assert len(agent_transcripts) == 1
        assert agent_transcripts[0].text == "I'm doing great, thanks!"


@pytest.mark.integration
class TestOpenAIRealtimeIntegration:
    """End-to-end tests against the live OpenAI Realtime API."""

    @pytest.fixture
    async def realtime(self):
        rt = Realtime(model="gpt-realtime", voice="alloy")
        rt.set_instructions("be friendly")
        try:
            await rt.connect()
            yield rt
        finally:
            await rt.close()

    async def test_simple_response_flow(self, realtime):
        async for _ in realtime.simple_response("Hello, can you hear me?"):
            pass

        await asyncio.sleep(3.0)
        audio = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAudioOutput)
        ]
        assert len(audio) > 0

    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        # Wait for connection to be fully established
        await asyncio.sleep(2.0)

        # OpenAI realtime expects 48 kHz PCM; resample from 16 kHz.
        samples_16k = mia_audio_16khz.samples
        num_samples_48k = int(len(samples_16k) * 48000 / 16000)
        samples_48k = signal.resample(samples_16k, num_samples_48k).astype(np.int16)

        audio_48khz = PcmData(
            samples=samples_48k, sample_rate=48000, format=AudioFormat.S16
        )

        async for _ in realtime.simple_response(
            "Listen to the following audio and tell me what you hear"
        ):
            pass
        await asyncio.sleep(5.0)

        await realtime.simple_audio_response(
            audio_48khz, Participant(original=None, user_id="u", id="u")
        )

        await asyncio.sleep(10.0)
        audio = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAudioOutput)
        ]
        assert len(audio) > 0

    async def test_video_sending_flow(self, realtime, bunny_video_track):
        async for _ in realtime.simple_response(
            "Describe what you see in this video please"
        ):
            pass
        await asyncio.sleep(10.0)
        await realtime.watch_video_track(bunny_video_track)

        await asyncio.sleep(10.0)

        await realtime.stop_watching_video_track()
        audio = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAudioOutput)
        ]
        assert len(audio) > 0
