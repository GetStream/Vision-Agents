import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv
from getstream.video.rtc import AudioFormat, PcmData
from google.genai.types import (
    Blob,
    Content,
    FunctionCall,
    LiveServerContent,
    LiveServerMessage,
    LiveServerSessionResumptionUpdate,
    LiveServerToolCall,
    Part,
    Transcription,
)
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeAudioOutputDoneEvent,
    RealtimeAudioOutputEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from vision_agents.core.tts.manual_test import play_pcm_with_ffplay
from vision_agents.plugins.gemini import Realtime
from vision_agents.plugins.gemini.gemini_realtime import GeminiRealtime

# Load environment variables
load_dotenv()


def _make_session(messages: list[LiveServerMessage]) -> AsyncMock:
    """Create a mock async session that yields the given messages."""
    session = AsyncMock()

    async def _receive():
        for msg in messages:
            yield msg

    session.receive = _receive
    return session


def _make_realtime() -> GeminiRealtime:
    """Create a GeminiRealtime instance without connecting."""
    with patch("vision_agents.plugins.gemini.gemini_realtime.genai"):
        rt = GeminiRealtime(api_key="fake-key")
    return rt


@pytest.fixture
async def realtime():
    """Create and manage Realtime connection lifecycle"""
    realtime = Realtime()
    try:
        yield realtime
    finally:
        await realtime.close()


class TestGeminiRealtime:
    """Integration tests for Gemini Realtime connect flow"""

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


class TestGeminiRealtimeProcessEvents:
    async def test_input_transcription(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                input_transcription=Transcription(text="hello"),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert len(emitted) == 1
        assert isinstance(emitted[0], RealtimeUserSpeechTranscriptionEvent)
        assert emitted[0].text == "hello"

    async def test_output_transcription(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                output_transcription=Transcription(text="world"),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert len(emitted) == 1
        assert isinstance(emitted[0], RealtimeAgentSpeechTranscriptionEvent)
        assert emitted[0].text == "world"

    async def test_model_turn_text(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[Part(text="response text")],
                ),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert len(emitted) == 1
        assert isinstance(emitted[0], LLMResponseChunkEvent)
        assert emitted[0].delta == "response text"

    async def test_model_turn_audio(self):
        rt = _make_realtime()
        audio_bytes = b"\x00" * 100
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[Part(inline_data=Blob(data=audio_bytes))],
                ),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert len(emitted) == 1
        assert isinstance(emitted[0], RealtimeAudioOutputEvent)

    async def test_model_turn_function_call(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                id="call_1", name="get_weather", args={"city": "NYC"}
                            )
                        )
                    ],
                ),
            ),
        )
        rt._real_session = _make_session([msg])
        rt._handle_function_call = AsyncMock()

        await rt._process_events()

        rt._handle_function_call.assert_called_once()
        call_arg = rt._handle_function_call.call_args[0][0]
        assert call_arg.name == "get_weather"

    async def test_turn_complete(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                turn_complete=True,
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert len(emitted) == 1
        assert isinstance(emitted[0], RealtimeAudioOutputDoneEvent)

    async def test_tool_call(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            tool_call=LiveServerToolCall(
                function_calls=[
                    FunctionCall(id="tc_1", name="search", args={"q": "test"})
                ],
            ),
        )
        rt._real_session = _make_session([msg])
        rt._handle_tool_calls = AsyncMock()

        await rt._process_events()

        rt._handle_tool_calls.assert_called_once_with(msg.tool_call)

    async def test_model_turn_with_turn_complete(self):
        """A single message can have both model_turn and turn_complete."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[Part(text="done")],
                ),
                turn_complete=True,
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        types = [type(e) for e in emitted]
        assert LLMResponseChunkEvent in types
        assert RealtimeAudioOutputDoneEvent in types

    async def test_part_with_text_and_audio(self):
        """A single Part with both text and inline_data must emit both events."""
        rt = _make_realtime()
        audio_bytes = b"\x00" * 100
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[Part(text="hello", inline_data=Blob(data=audio_bytes))],
                ),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        types = [type(e) for e in emitted]
        assert LLMResponseChunkEvent in types
        assert RealtimeAudioOutputEvent in types

    async def test_transcription_with_audio_same_message(self):
        """Audio in model_turn must not be skipped when transcription is also present."""
        rt = _make_realtime()
        audio_bytes = b"\x00" * 100
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                output_transcription=Transcription(text="hi"),
                model_turn=Content(
                    parts=[Part(inline_data=Blob(data=audio_bytes))],
                ),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        types = [type(e) for e in emitted]
        assert RealtimeAgentSpeechTranscriptionEvent in types
        assert RealtimeAudioOutputEvent in types

    async def test_thought_only_parts_not_logged_as_unrecognized(self, caplog):
        """model_turn with only thought parts should still be handled."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[Part(text="thinking...", thought=True)],
                ),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        with caplog.at_level("DEBUG"):
            await rt._process_events()

        assert "Unrecognized" not in caplog.text

    async def test_empty_parts_not_logged_as_unrecognized(self, caplog):
        """model_turn with empty parts should still be handled."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(parts=[]),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        with caplog.at_level("DEBUG"):
            await rt._process_events()

        assert "Unrecognized" not in caplog.text

    async def test_unrecognized_message_logged(self, caplog):
        """A message with no recognized fields should be logged."""
        rt = _make_realtime()
        msg = LiveServerMessage()
        rt._real_session = _make_session([msg])

        with caplog.at_level("DEBUG"):
            await rt._process_events()

        assert "Unrecognized" in caplog.text

    async def test_session_resumption_update(self):
        """Session resumption handle is stored when present on a response."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(parts=[Part(text="hi")]),
            ),
            session_resumption_update=LiveServerSessionResumptionUpdate(
                resumable=True,
                new_handle="resume-token-123",
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert rt._session_resumption_id == "resume-token-123"

    async def test_input_and_output_transcription_same_message(self):
        """Both input and output transcription in same message are both handled."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                input_transcription=Transcription(text="user said"),
                output_transcription=Transcription(text="model said"),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        types = [type(e) for e in emitted]
        assert RealtimeUserSpeechTranscriptionEvent in types
        assert RealtimeAgentSpeechTranscriptionEvent in types

    async def test_empty_transcription_text_not_emitted(self):
        """Transcription with empty text should not emit events."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                input_transcription=Transcription(text=""),
                output_transcription=Transcription(text=""),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert len(emitted) == 0

    async def test_begin_response_called_for_model_turn(self):
        """_begin_response should be called when model_turn is present."""
        rt = _make_realtime()
        rt._epoch = 5
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(parts=[Part(text="hi")]),
            ),
        )
        rt._real_session = _make_session([msg])

        emitted: list[object] = []
        rt.events.send = lambda e: emitted.append(e)

        await rt._process_events()

        assert rt._response_epoch == 5

    async def test_function_call_does_not_block_event_loop(self):
        """_process_events must not block while a function call executes."""
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                model_turn=Content(
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                id="call_1", name="slow_tool", args={}
                            )
                        )
                    ],
                ),
            ),
        )
        rt._real_session = _make_session([msg])

        async def slow_handle(function_call: FunctionCall, timeout: float = 30.0):
            await asyncio.sleep(10)

        rt._handle_function_call = slow_handle

        try:
            finished = await asyncio.wait_for(rt._process_events(), timeout=2.0)
            assert finished is False
        finally:
            await rt.close()
