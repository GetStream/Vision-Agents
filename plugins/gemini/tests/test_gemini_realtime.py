import asyncio
from types import SimpleNamespace
from typing import cast

import av
import numpy as np
import pytest
from google.genai.live import AsyncSession
from dotenv import load_dotenv
from google.genai.errors import APIError
from google.genai.types import Blob
from getstream.video.rtc import AudioFormat, PcmData
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeAudioOutputDoneEvent,
    RealtimeAudioOutputEvent,
)
from vision_agents.core.tts.manual_test import play_pcm_with_ffplay
from vision_agents.plugins.gemini import Realtime
from vision_agents.plugins.gemini.gemini_realtime import _should_reconnect

# Load environment variables
load_dotenv()


def _solid_color_frame() -> av.VideoFrame:
    frame_array = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_array[:, :] = [255, 0, 0]
    return av.VideoFrame.from_ndarray(frame_array, format="rgb24")


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _ReceiveSession:
    def __init__(self, messages: list[object]) -> None:
        self._messages = messages

    async def receive(self):
        for message in self._messages:
            yield message


class _FatalProcessingRealtime(Realtime):
    def __init__(self) -> None:
        super().__init__()
        self.process_calls = 0

    async def _process_events(self) -> bool:
        self.process_calls += 1
        raise APIError(
            1007,
            {
                "message": (
                    "realtime_input.media_chunks is deprecated. "
                    "Use audio, video, or text instead."
                )
            },
            None,
        )


def _server_message(
    *,
    output_text: str | None = None,
    turn_complete: bool = False,
):
    return SimpleNamespace(
        server_content=SimpleNamespace(
            input_transcription=None,
            output_transcription=(
                SimpleNamespace(text=output_text) if output_text is not None else None
            ),
            model_turn=None,
            turn_complete=turn_complete,
        ),
        session_resumption_update=None,
        tool_call=None,
    )


@pytest.fixture
async def realtime():
    """Create and manage Realtime connection lifecycle"""
    realtime = Realtime()
    try:
        yield realtime
    finally:
        await realtime.close()


class TestGeminiRealtime:
    """Tests for Gemini Realtime connect flow."""

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime):
        """Test sending a simple text message and receiving response"""
        # Send a simple message
        events = []
        pcm = PcmData(sample_rate=24000, format=AudioFormat.S16)

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)
            pcm.append(event.data)

        await asyncio.sleep(0.01)
        await realtime.connect()
        await realtime.simple_response("Hello, can you hear me?")

        # Wait for response
        await asyncio.sleep(3.0)
        assert len(events) > 0

        # play the generated audio
        await play_pcm_with_ffplay(pcm)

    @pytest.mark.integration
    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        """Test sending real audio data and verify connection remains stable"""
        events = []

        @realtime.events.subscribe
        async def on_audio(event: RealtimeAudioOutputEvent):
            events.append(event)

        await asyncio.sleep(0.01)
        await realtime.connect()

        await realtime.simple_response(
            "Listen to the following story, what is Mia looking for?"
        )
        await asyncio.sleep(10.0)
        await realtime.simple_audio_response(mia_audio_16khz)

        # Wait a moment to ensure processing
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
        await realtime.watch_video_track(bunny_video_track)

        # Let it run for a few seconds
        await asyncio.sleep(10.0)

        # Stop video sender
        await realtime.stop_watching_video_track()
        assert len(events) > 0

    async def test_send_video_frame_uses_video_input(self):
        """Test that Gemini sends video frames using the video field."""
        realtime = Realtime()
        session = _RecordingSession()
        realtime._real_session = cast(AsyncSession, session)
        realtime.connected = True

        try:
            await realtime._send_video_frame(_solid_color_frame())
        finally:
            await realtime.close()

        assert len(session.calls) == 1
        sent_payload = session.calls[0]
        assert "video" in sent_payload
        assert "media" not in sent_payload

        sent_blob = cast(Blob, sent_payload["video"])
        assert sent_blob.mime_type == "image/png"
        assert sent_blob.data

    async def test_processing_loop_stops_on_fatal_api_error(self):
        """Test that fatal Gemini API errors do not loop forever."""
        realtime = _FatalProcessingRealtime()
        realtime.connected = True

        task = asyncio.create_task(realtime._processing_loop())
        await asyncio.wait_for(task, timeout=1.0)

        assert realtime.process_calls == 1
        assert not realtime.connected

        await realtime.close()

    def test_should_reconnect_for_transient_api_error(self):
        """Test reconnect behavior for transient Gemini websocket close codes."""
        transient_error = APIError(1011, {"message": "session timeout"}, None)
        fatal_error = APIError(
            1007,
            {"message": "realtime_input.media_chunks is deprecated"},
            None,
        )

        assert _should_reconnect(transient_error)
        assert not _should_reconnect(fatal_error)

    async def test_turn_complete_finalizes_agent_transcript(self):
        """Test that Gemini finalizes transcript state when a turn completes."""
        realtime = Realtime()
        realtime._real_session = cast(
            AsyncSession,
            _ReceiveSession(
                [
                    _server_message(output_text="Hello "),
                    _server_message(output_text="there"),
                    _server_message(turn_complete=True),
                ]
            ),
        )

        agent_transcripts: list[RealtimeAgentSpeechTranscriptionEvent] = []
        audio_done_events: list[RealtimeAudioOutputDoneEvent] = []

        @realtime.events.subscribe
        async def on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent):
            agent_transcripts.append(event)

        @realtime.events.subscribe
        async def on_audio_done(event: RealtimeAudioOutputDoneEvent):
            audio_done_events.append(event)

        await realtime._process_events()

        assert [(event.text, event.mode) for event in agent_transcripts] == [
            ("Hello ", "delta"),
            ("there", "delta"),
            ("", "final"),
        ]
        assert len(audio_done_events) == 1
