import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import websockets
from dotenv import load_dotenv
from google.genai.errors import APIError
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
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.core.llm.realtime import (
    RealtimeAgentSpeechEnded,
    RealtimeAgentSpeechStarted,
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeAudioOutputDone,
    RealtimeUserSpeechEnded,
    RealtimeUserSpeechStarted,
    RealtimeUserTranscript,
)
from vision_agents.plugins.gemini import Realtime
from vision_agents.plugins.gemini.gemini_realtime import GeminiRealtime
from websockets.frames import Close

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


class TestGeminiRealtimeProcessEvents:
    async def test_input_transcription(self):
        rt = _make_realtime()
        rt._current_participant = Participant(original=None, user_id="u", id="u")
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                input_transcription=Transcription(text="hello"),
            ),
        )
        rt._real_session = _make_session([msg])

        await rt._process_events()

        items = rt.output.peek()
        assert len(items) == 1
        assert isinstance(items[0], RealtimeUserTranscript)
        assert items[0].text == "hello"

    async def test_output_transcription(self):
        rt = _make_realtime()
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                output_transcription=Transcription(text="world"),
            ),
        )
        rt._real_session = _make_session([msg])

        await rt._process_events()

        items = rt.output.peek()
        assert len(items) == 1
        assert isinstance(items[0], RealtimeAgentTranscript)
        assert items[0].text == "world"

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

        await rt._process_events()

        items = rt.output.peek()
        assert len(items) == 1
        assert isinstance(items[0], RealtimeAudioOutput)

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

        await rt._process_events()

        items = rt.output.peek()
        assert len(items) == 1
        assert isinstance(items[0], RealtimeAudioOutputDone)

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

        assert any(isinstance(e, LLMResponseChunkEvent) for e in emitted)
        assert any(isinstance(i, RealtimeAudioOutputDone) for i in rt.output.peek())

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

        assert any(isinstance(e, LLMResponseChunkEvent) for e in emitted)
        assert any(isinstance(i, RealtimeAudioOutput) for i in rt.output.peek())

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

        await rt._process_events()

        types = [type(i) for i in rt.output.peek()]
        assert RealtimeAgentTranscript in types
        assert RealtimeAudioOutput in types

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

        await rt._process_events()

        assert rt._session_resumption_id == "resume-token-123"

    async def test_input_and_output_transcription_same_message(self):
        """Both input and output transcription in same message are both handled."""
        rt = _make_realtime()
        rt._current_participant = Participant(original=None, user_id="u", id="u")
        msg = LiveServerMessage(
            server_content=LiveServerContent(
                input_transcription=Transcription(text="user said"),
                output_transcription=Transcription(text="model said"),
            ),
        )
        rt._real_session = _make_session([msg])

        await rt._process_events()

        types = [type(i) for i in rt.output.peek()]
        assert RealtimeUserTranscript in types
        assert RealtimeAgentTranscript in types

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

        await rt._process_events()

        assert len(rt.output.peek()) == 0

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


class TestGeminiRealtimeProcessingLoop:
    async def test_clean_websocket_close_marks_disconnected(self):
        rt = _make_realtime()
        rt.connected = True

        async def _raise_closed_ok():
            raise websockets.ConnectionClosedOK(Close(1000, "normal"), None)

        rt._process_events = _raise_closed_ok

        await rt._processing_loop()

        assert rt.connected is False

    async def test_non_reconnectable_close_marks_disconnected(self):
        rt = _make_realtime()
        rt.connected = True

        async def _raise_closed_error():
            raise websockets.ConnectionClosedError(Close(1002, "protocol error"), None)

        rt._process_events = _raise_closed_error

        await rt._processing_loop()

        assert rt.connected is False

    async def test_reconnectable_close_triggers_reconnect(self):
        rt = _make_realtime()
        reconnected = asyncio.Event()

        async def _raise_reconnectable():
            if not reconnected.is_set():
                reconnected.set()
                raise websockets.ConnectionClosedError(Close(1011, "timeout"), None)
            await asyncio.sleep(10)

        rt._process_events = _raise_reconnectable
        rt._establish_session = AsyncMock()

        task = asyncio.create_task(rt._processing_loop())
        await asyncio.wait_for(reconnected.wait(), timeout=2.0)
        task.cancel()
        await task

        rt._establish_session.assert_called_once()

    async def test_consecutive_errors_stop_loop(self):
        rt = _make_realtime()
        rt.connected = True

        async def _always_fail():
            raise ConnectionError("network down")

        rt._process_events = _always_fail

        await rt._processing_loop()

        assert rt.connected is False

    async def test_api_error_with_reconnectable_code_triggers_reconnect(self):
        rt = _make_realtime()
        reconnected = asyncio.Event()

        async def _raise_api_error():
            if not reconnected.is_set():
                reconnected.set()
                raise APIError(1011, None, None)
            await asyncio.sleep(10)

        rt._process_events = _raise_api_error
        rt._establish_session = AsyncMock()

        task = asyncio.create_task(rt._processing_loop())
        await asyncio.wait_for(reconnected.wait(), timeout=2.0)
        task.cancel()
        await task

        rt._establish_session.assert_called_once()

    async def test_api_error_non_reconnectable_stops_loop(self):
        rt = _make_realtime()
        rt.connected = True

        async def _raise_api_error():
            raise APIError(1002, None, None)

        rt._process_events = _raise_api_error

        await rt._processing_loop()

        assert rt.connected is False


class TestGeminiRealtimeFunctionCalling:
    """Unit tests for Gemini Realtime tool conversion and config wiring."""

    async def test_convert_tools_to_provider_format(self):
        """Test tool conversion to Gemini Live format."""
        # Create a minimal instance just for testing the conversion method
        realtime = _make_realtime()

        # Test tools
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        }
                    },
                    "required": ["expression"],
                },
            },
        ]

        result = realtime._convert_tools_to_provider_format(tools)

        assert len(result) == 1
        assert "function_declarations" in result[0]
        assert len(result[0]["function_declarations"]) == 2

        # Check first tool
        tool1 = result[0]["function_declarations"][0]
        assert tool1["name"] == "get_weather"
        assert tool1["description"] == "Get weather information"
        assert "location" in tool1["parameters"]["properties"]

        # Check second tool
        tool2 = result[0]["function_declarations"][1]
        assert tool2["name"] == "calculate"
        assert tool2["description"] == "Perform calculations"
        assert "expression" in tool2["parameters"]["properties"]

    async def test_get_config_with_tools(self):
        """Test that tools are added to the config."""
        # Create a minimal instance for testing config creation
        realtime = _make_realtime()

        # Register a test function
        @realtime.register_function(description="Test function")
        async def test_func(param: str) -> str:
            return f"test: {param}"

        config = realtime.get_config()

        # Verify tools were added
        assert "tools" in config
        assert len(config["tools"]) == 1
        assert "function_declarations" in config["tools"][0]
        assert len(config["tools"][0]["function_declarations"]) == 1
        assert config["tools"][0]["function_declarations"][0]["name"] == "test_func"

    async def test_get_config_without_tools(self):
        """Test config creation when no tools are available."""
        # Create a minimal instance without registering any functions
        realtime = _make_realtime()

        config = realtime.get_config()

        # Verify tools were not added
        assert "tools" not in config


@pytest.fixture
async def realtime():
    rt = Realtime()
    try:
        await rt.connect()
        yield rt
    finally:
        await rt.close()


@pytest.fixture
async def realtime_with_tools():
    """Realtime with all function-calling test tools registered, then connected.

    Yields ``(rt, function_calls)`` where ``function_calls`` is a list each
    registered function appends to when invoked by the model.
    """
    rt = Realtime()
    function_calls: list[dict[str, Any]] = []

    @rt.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> dict[str, str]:
        """Get weather information for a location."""
        function_calls.append({"name": "get_weather", "location": location})
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%",
        }

    @rt.register_function(description="A function that sometimes fails")
    async def unreliable_function(input_data: str) -> dict[str, Any]:
        """A function that raises an error for testing."""
        function_calls.append({"name": "unreliable_function", "input": input_data})
        if "error" in input_data.lower():
            raise ValueError("Simulated error for testing")
        return {"result": f"Success: {input_data}"}

    @rt.register_function(description="Get current time")
    async def get_time() -> dict[str, str]:
        """Get current time."""
        function_calls.append({"name": "get_time"})
        return {"time": "2024-01-15 14:30:00", "timezone": "UTC"}

    @rt.register_function(description="Get system status")
    async def get_status() -> dict[str, str]:
        """Get system status."""
        function_calls.append({"name": "get_status"})
        return {"status": "healthy", "uptime": "24h"}

    try:
        await rt.connect()
        yield rt, function_calls
    finally:
        await rt.close()


@pytest.mark.integration
class TestGeminiRealtimeIntegration:
    """End-to-end tests against the live Gemini Realtime API."""

    async def test_simple_response_flow(self, realtime):
        async for _ in realtime.simple_response("Hello, can you hear me?"):
            pass

        await asyncio.sleep(3.0)
        items = realtime.output.peek()
        audio = [i for i in items if isinstance(i, RealtimeAudioOutput)]
        agent_started = [
            i for i in items if isinstance(i, RealtimeAgentSpeechStarted)
        ]
        agent_ended = [i for i in items if isinstance(i, RealtimeAgentSpeechEnded)]
        assert len(audio) > 0
        assert len(agent_started) >= 1
        assert len(agent_ended) >= 1
        assert any(not e.interrupted for e in agent_ended)

    async def test_audio_sending_flow(self, realtime, mia_audio_16khz):
        async for _ in realtime.simple_response(
            "Listen to the following story, what is Mia looking for?"
        ):
            pass
        await asyncio.sleep(10.0)
        await realtime.simple_audio_response(mia_audio_16khz)

        await asyncio.sleep(10.0)
        items = realtime.output.peek()
        audio = [i for i in items if isinstance(i, RealtimeAudioOutput)]
        user_started = [
            i for i in items if isinstance(i, RealtimeUserSpeechStarted)
        ]
        user_ended = [i for i in items if isinstance(i, RealtimeUserSpeechEnded)]
        assert len(audio) > 0
        assert len(user_started) >= 1
        assert len(user_ended) >= 1

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

    async def test_live_function_calling_basic(self, realtime_with_tools):
        rt, function_calls = realtime_with_tools

        async for _ in rt.simple_response(
            "What's the weather like in New York? Please use the get_weather function to check."
        ):
            pass
        await asyncio.sleep(8.0)

        weather_calls = [c for c in function_calls if c["name"] == "get_weather"]
        assert len(weather_calls) > 0, "get_weather was not called by Gemini"
        assert weather_calls[0]["location"] == "New York"

    async def test_live_function_calling_error_handling(self, realtime_with_tools):
        rt, function_calls = realtime_with_tools

        async for _ in rt.simple_response(
            "Please call the unreliable_function with 'error test' as input."
        ):
            pass
        await asyncio.sleep(8.0)

        unreliable_calls = [
            c for c in function_calls if c["name"] == "unreliable_function"
        ]
        assert len(unreliable_calls) > 0, "unreliable_function was not called"
        items = rt.output.peek()
        assert any(
            isinstance(i, (RealtimeAudioOutput, RealtimeAgentTranscript)) for i in items
        ), "No agent output received after function error"

    async def test_live_function_calling_multiple_functions(self, realtime_with_tools):
        rt, function_calls = realtime_with_tools

        async for _ in rt.simple_response(
            "Please check the current time and system status using the available functions."
        ):
            pass
        await asyncio.sleep(10.0)

        function_names = [call["name"] for call in function_calls]
        assert "get_time" in function_names, "get_time function was not called"
        assert "get_status" in function_names, "get_status function was not called"
