"""Unit tests for TestSession event capture, overlap safety and Agent support."""

import asyncio
from typing import Any

import pytest
from getstream.video.rtc import AudioStreamTrack

from tests.test_testing.fake_llms import ToolCallingLLM
from tests.test_testing.fakes import FakeMCPServer
from vision_agents.core import Agent, User
from vision_agents.core.edge import EdgeTransport
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import ToolEndEvent, ToolStartEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem
from vision_agents.core.tts import TTS
from vision_agents.testing import (
    ChatMessageEvent,
    FunctionCallOutputEvent,
    TestSession,
)


def _call(name: str, arguments: dict[str, Any], call_id: str) -> NormalizedToolCallItem:
    return {
        "type": "tool_call",
        "name": name,
        "arguments_json": arguments,
        "id": call_id,
    }


class _FakeCall:
    def __init__(self, call_id: str) -> None:
        self.id = call_id


class _FakeEdge(EdgeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.events = EventManager()

    async def authenticate(self, user: User) -> None:
        pass

    async def create_call(self, call_id: str, **kwargs: Any) -> Any:
        return _FakeCall(call_id)

    def create_audio_track(self) -> AudioStreamTrack:
        return AudioStreamTrack(
            audio_buffer_size_ms=300_000, sample_rate=48000, channels=2
        )

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def join(self, agent: Any, call: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def publish_tracks(self, audio_track: Any, video_track: Any) -> None:
        pass

    async def create_conversation(self, call: Any, user: User, instructions: str):
        return None

    def add_track_subscriber(self, track_id: str) -> None:
        return None

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        pass


class _FakeTTS(TTS):
    model = "tts"

    async def stream_audio(self, *args: Any, **kwargs: Any) -> bytes:
        return b""

    async def stop_audio(self) -> None:
        pass


@pytest.fixture
def weather_llm() -> ToolCallingLLM:
    llm = ToolCallingLLM(
        script={"weather": [_call("get_weather", {"location": "Berlin"}, "c1")]},
        reply="Sunny.",
    )

    @llm.register_function(description="weather tool")
    async def get_weather(location: str) -> dict:
        return {"temp": 70, "location": location}

    return llm


@pytest.fixture
def agent(weather_llm: ToolCallingLLM) -> Agent:
    return Agent(
        edge=_FakeEdge(),
        llm=weather_llm,
        tts=_FakeTTS(),
        agent_user=User(name="test"),
        instructions="Answer in one word.",
    )


class TestTestSession:
    async def test_captures_call_output_and_messages_in_order(
        self, weather_llm: ToolCallingLLM
    ):
        async with TestSession(llm=weather_llm) as session:
            response = await session.simple_response("weather")

        assert [type(e).__name__ for e in response.events] == [
            "ChatMessageEvent",
            "FunctionCallEvent",
            "FunctionCallOutputEvent",
            "ChatMessageEvent",
        ]
        assert response.events[0] == ChatMessageEvent(role="user", content="weather")
        assert response.output == "Sunny."
        assert response.function_calls[0].tool_call_id == "c1"
        response.assert_function_called("get_weather", arguments={"location": "Berlin"})
        response.assert_function_output(
            "get_weather", output={"temp": 70, "location": "Berlin"}
        )
        output = response.events[2]
        assert isinstance(output, FunctionCallOutputEvent)
        assert output.tool_call_id == "c1"
        assert output.execution_time_ms is not None

    async def test_parallel_calls_of_same_tool_are_paired_by_tool_call_id(self):
        llm = ToolCallingLLM(
            script={
                "both": [
                    _call("get_weather", {"location": "Berlin"}, "c1"),
                    _call("get_weather", {"location": "Tokyo"}, "c2"),
                ]
            }
        )

        @llm.register_function(description="weather tool")
        async def get_weather(location: str) -> str:
            # Berlin finishes last so outputs arrive out of call order.
            await asyncio.sleep(0.05 if location == "Berlin" else 0)
            return f"weather:{location}"

        async with TestSession(llm=llm) as session:
            response = await session.simple_response("both")

        calls = {fc.tool_call_id: fc for fc in response.function_calls}
        outputs = {
            e.tool_call_id: e
            for e in response.events
            if isinstance(e, FunctionCallOutputEvent)
        }
        assert calls["c1"].arguments == {"location": "Berlin"}
        assert calls["c2"].arguments == {"location": "Tokyo"}
        assert outputs["c1"].output == "weather:Berlin"
        assert outputs["c2"].output == "weather:Tokyo"
        output_order = [
            e.tool_call_id
            for e in response.events
            if isinstance(e, FunctionCallOutputEvent)
        ]
        assert output_order == ["c2", "c1"]

    async def test_tool_error_is_captured(self):
        llm = ToolCallingLLM(script={"fail": [_call("explode", {}, "c1")]})

        @llm.register_function(description="always fails")
        async def explode() -> None:
            raise RuntimeError("boom")

        async with TestSession(llm=llm) as session:
            response = await session.simple_response("fail")

        response.assert_function_output(
            "explode", output={"error": "boom"}, is_error=True
        )

    async def test_overlapping_calls_keep_events_separate_and_registry_intact(
        self,
    ):
        llm = ToolCallingLLM(
            script={
                "a": [_call("tool_a", {}, "a1")],
                "b": [_call("tool_b", {}, "b1")],
            }
        )

        @llm.register_function(description="a")
        async def tool_a() -> str:
            await asyncio.sleep(0.05)
            return "A"

        @llm.register_function(description="b")
        async def tool_b() -> str:
            await asyncio.sleep(0.01)
            return "B"

        originals = {
            name: fd.function for name, fd in llm.function_registry.functions.items()
        }

        async with TestSession(llm=llm) as session:
            response_a, response_b = await asyncio.gather(
                session.simple_response("a"), session.simple_response("b")
            )

        assert [fc.name for fc in response_a.function_calls] == ["tool_a"]
        assert [fc.name for fc in response_b.function_calls] == ["tool_b"]
        response_a.assert_function_output("tool_a", output="A")
        response_b.assert_function_output("tool_b", output="B")
        for name, fd in llm.function_registry.functions.items():
            assert fd.function is originals[name]

    async def test_transcript_accumulates_across_turns(
        self, weather_llm: ToolCallingLLM
    ):
        async with TestSession(llm=weather_llm) as session:
            await session.simple_response("hello")
            await session.simple_response("weather")
            transcript = session.transcript

        roles = [e.role for e in transcript if isinstance(e, ChatMessageEvent)]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert [e.name for e in transcript if not isinstance(e, ChatMessageEvent)] == [
            "get_weather",
            "get_weather",
        ]

    async def test_empty_reply_has_no_assistant_message(self):
        llm = ToolCallingLLM(reply="")
        async with TestSession(llm=llm) as session:
            response = await session.simple_response("hi")

        assert response.output is None
        assert response.events == [ChatMessageEvent(role="user", content="hi")]

    async def test_mocked_tool_output_is_captured(self, weather_llm: ToolCallingLLM):
        async with TestSession(llm=weather_llm) as session:
            with session.mock_functions({"get_weather": lambda **_: {"temp": 99}}):
                response = await session.simple_response("weather")

        response.assert_function_output("get_weather", output={"temp": 99})

    async def test_close_unsubscribes_tool_events(self, weather_llm: ToolCallingLLM):
        async with TestSession(llm=weather_llm):
            assert weather_llm.events.has_subscribers(ToolStartEvent)

        assert not weather_llm.events.has_subscribers(ToolStartEvent)
        assert not weather_llm.events.has_subscribers(ToolEndEvent)

    async def test_default_instructions(self, weather_llm: ToolCallingLLM):
        async with TestSession(llm=weather_llm) as session:
            assert session.instructions == "You are a helpful assistant."

    async def test_simple_response_requires_start(self, weather_llm: ToolCallingLLM):
        session = TestSession(llm=weather_llm)
        with pytest.raises(RuntimeError, match="not started"):
            await session.simple_response("hi")

    def test_requires_llm_or_agent(self):
        with pytest.raises(ValueError, match="either 'llm' or 'agent'"):
            TestSession()

    def test_rejects_llm_together_with_agent(
        self, weather_llm: ToolCallingLLM, agent: Agent
    ):
        with pytest.raises(ValueError, match="not both"):
            TestSession(llm=weather_llm, agent=agent)

    def test_rejects_instructions_together_with_agent(self, agent: Agent):
        with pytest.raises(ValueError, match="instructions"):
            TestSession(instructions="x", agent=agent)

    async def test_agent_mode_uses_agent_llm_and_instructions(
        self, weather_llm: ToolCallingLLM, agent: Agent
    ):
        async with TestSession(agent=agent) as session:
            assert session.llm is weather_llm
            assert session.instructions == "Answer in one word."
            response = await session.simple_response("weather")

        response.assert_function_called("get_weather", arguments={"location": "Berlin"})
        assert response.output == "Sunny."

    async def test_agent_mode_connects_mcp_servers_and_captures_their_tools(self):
        server = FakeMCPServer()
        llm = ToolCallingLLM(
            script={"find": [_call("mcp_0_lookup", {"key": "42"}, "c1")]},
            reply="Found it.",
        )
        agent = Agent(
            edge=_FakeEdge(),
            llm=llm,
            tts=_FakeTTS(),
            agent_user=User(name="test"),
            mcp_servers=[server],
        )

        async with TestSession(agent=agent) as session:
            assert server.is_connected
            response = await session.simple_response("find")

        assert not server.is_connected
        assert server.calls == [("lookup", {"key": "42"})]
        response.assert_function_called("mcp_0_lookup", arguments={"key": "42"})
        response.assert_function_output("mcp_0_lookup", output="record 42")
